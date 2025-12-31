from typing import cast, List, Union
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoConfig, AutoTokenizer, is_torch_npu_available, MistralConfig, MistralForCausalLM
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
import os
from peft import PeftModel
import re
import sys
sys.path.append('../finetune')
from bidirectional_mistral import MistralBiForCausalLM


def find_largest_checkpoint(checkpoint_dir):
    checkpoint_pattern = re.compile(r'checkpoint-(\d+)')
    max_number = -1
    max_checkpoint_file = None
    for file in os.listdir(checkpoint_dir):
        match = checkpoint_pattern.search(file)
        if match:
            number = int(match.group(1))
            if number > max_number:
                max_number = number
                max_checkpoint_file = file
    if max_checkpoint_file:
        return os.path.join(checkpoint_dir, max_checkpoint_file)
    else:
        return None

def pooling_func(vecs: Tensor, attention_mask: Tensor, pooling_method: str) -> Tensor:
    if pooling_method == 'cls-pooling':
        return torch.log(1 + torch.relu(vecs[:, -1, :]))
    elif pooling_method == 'max-pooling':
        values, _ = torch.max(torch.log(1 + torch.relu(vecs)) * attention_mask.unsqueeze(-1), dim=1)
        return values
    elif pooling_method == 'sum-pooling':
        return torch.sum(torch.log(1 + torch.relu(vecs)) * attention_mask.unsqueeze(-1), dim=1)
    else:
        raise ValueError(f"Invalid pooling method: {pooling_method}")


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'<instruct>{task_description}\n<query>{query}'


class LENSModel:
    def __init__(
        self,
        base_model_name_or_path: str = 'mistralai/Mistral-7B-v0.1',
        model_name_or_path: str = None,
        normalize_embeddings: bool = True,
        query_instruction_for_retrieval: str = 'Given a query, retrieval relevant passages that answer the query.',
        use_fp16: bool = True,
        pooling_method: str = 'max-pooling',
        bidirectional: bool = False,
        cache_dir: str = None,
    ) -> None:
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            if self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
                self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        special_tokens_dict = {'additional_special_tokens': ['<instruct>', '<query>', '<response>']}
        add_num = self.tokenizer.add_special_tokens(special_tokens_dict)

        # Load the model
        if model_name_or_path:
            config = AutoConfig.from_pretrained(base_model_name_or_path, cache_dir=cache_dir)
        else:
            raise ValueError(
                "You are instantiating a new config instance from scratch. This is not supported by this script."
            )
        config.use_cache = False

        if base_model_name_or_path:
            if bidirectional:
                print("Using bidirectional model")
                self.model = MistralBiForCausalLM.from_pretrained(
                    base_model_name_or_path,
                    use_flash_attention_2=True,
                    cache_dir=cache_dir,
                    from_tf=bool(".ckpt" in base_model_name_or_path),
                    config=config,
            )
            else:
                print("Using unidirectional model")
                self.model = MistralForCausalLM.from_pretrained(
                    base_model_name_or_path,
                    use_flash_attention_2=True,
                    cache_dir=cache_dir,
                    from_tf=bool(".ckpt" in base_model_name_or_path),
                    config=config,
                )
        else:
            raise ValueError("Base model name or path is required")

        if model_name_or_path is not None:
            # Load the input embeddings and the lm head
            input_embeddings = torch.load(os.path.join(model_name_or_path, 'embedding', 'input_emb.pth'))
            lm_head = torch.load(os.path.join(model_name_or_path, 'embedding', 'lm_head.pth'))
            self.model.set_input_embeddings(input_embeddings)
            self.model.lm_head = lm_head
        
            # Load the lora module
            print(f"Loading Lora from {model_name_or_path}")
            self.model = PeftModel.from_pretrained(self.model, find_largest_checkpoint(model_name_or_path))
            self.model = self.model.merge_and_unload()

        print(self.model)

        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.pooling_method = pooling_method

        self.suffix = "</s>" 

        self.normalize_embeddings = normalize_embeddings

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif is_torch_npu_available():
            self.device = torch.device("npu")
        else:
            self.device = torch.device("cpu")
            use_fp16 = False
        if use_fp16: self.model.half()
        self.model = self.model.to(self.device)

        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            print(f"----------using {self.num_gpus}*GPUs----------")
            self.model = torch.nn.DataParallel(self.model)

    def set_instruction(self, query_instruction_for_retrieval: str):
        self.query_instruction_for_retrieval = query_instruction_for_retrieval

    @torch.no_grad()
    def encode_queries(self, queries: Union[List[str], str],
                       batch_size: int = 256,
                       max_length: int = None) -> np.ndarray:
        self.model.eval()
        '''
        This function will be used for retrieval task
        if there is a instruction for queries, we will add it to the query text
        '''
        if isinstance(queries, str):
            sentences = [get_detailed_instruct(self.query_instruction_for_retrieval, queries)]
        else:
            sentences = [get_detailed_instruct(self.query_instruction_for_retrieval, q) for q in queries]

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in tqdm(range(0, len(sentences_sorted), batch_size), desc="Inference Embeddings",
                                disable=len(sentences_sorted) < 256):
            sentences_batch = sentences_sorted[start_index:start_index + batch_size]
            for i in range(len(sentences_batch)):
                sentences_batch[i] = sentences_batch[i] + self.suffix
            inputs = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=max_length,
                add_special_tokens=True
            ).to(self.device)

            outputs = self.model(**inputs, return_dict=True)
            pooling_mask = torch.zeros_like(inputs['input_ids'], dtype=inputs['attention_mask'].dtype)
            special_token_id = self.tokenizer.convert_tokens_to_ids('<query>')
            for idx, seq in enumerate(inputs['input_ids']):
                special_pos = (seq == special_token_id).nonzero()
                if len(special_pos) > 0:
                    last_pos = special_pos[-1].item()
                    pooling_mask[idx, last_pos:-2] = 1
                else:
                    raise ValueError("No special token found")
            embeddings = pooling_func(outputs.logits, pooling_mask, self.pooling_method)

            if self.normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.extend(embeddings.float().cpu())

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        return all_embeddings

    @torch.no_grad()
    def encode_corpus(self,
                      corpus: Union[List[str], str],
                      batch_size: int = 256,
                      max_length: int = None) -> np.ndarray:
        '''
        This function will be used for retrieval task
        encode corpus for retrieval task
        '''
        self.model.eval()

        if isinstance(corpus, str):
            sentences = [corpus]
        else:
            sentences = corpus

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in tqdm(range(0, len(sentences_sorted), batch_size), desc="Inference Embeddings",
                                disable=len(sentences_sorted) < 256):
            sentences_batch = sentences_sorted[start_index:start_index + batch_size]
            for i in range(len(sentences_batch)):
                sentences_batch[i] = sentences_batch[i] + self.suffix
            inputs = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=max_length,
                add_special_tokens=True
            ).to(self.device)

            outputs = self.model(**inputs, return_dict=True)
            # Set last two positions to 0 because of autoregressive decoding
            inputs['attention_mask'][:, -2:] = 0
            embeddings = pooling_func(outputs.logits, inputs['attention_mask'], self.pooling_method)

            if self.normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.extend(embeddings.float().cpu())

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        return all_embeddings

    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):  # Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])  # Sum of length of individual strings
# LENS: Enhancing Lexicon-Based Text Embeddings with Large Language Models

## Introduction ##
This is the official code for ACL 2025 paper "Enhancing Lexicon-Based Text Embeddings with Large Language Models".

## Environment
```
conda create -n lens_310 python=3.10
conda activate lens_310

conda install pytorch pytorch-cuda=12.6 -c pytorch -c nvidia
pip install transformers==4.43.1 deepspeed accelerate datasets peft pandas
pip install flash-attn --no-build-isolation
```

## Model Checkpoints
| Models |  Link  |
|---|---|
|LENS-d4000  | [yibinlei/LENS-d4000](https://huggingface.co/yibinlei/LENS-d4000) | 
|LENS-d8000 | [yibinlei/LENS-d8000](https://huggingface.co/yibinlei/LENS-d8000) | 

You can use them with:
```
import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoTokenizer
from bidirectional_mistral import MistralBiForCausalLM

def get_detailed_instruct(task_instruction: str, query: str) -> str:
    return f'<instruct>{task_instruction}\n<query>{query}'

def pooling_func(vecs: Tensor, pooling_mask: Tensor) -> Tensor:
    # We use max-pooling for LENS.
    return torch.max(torch.log(1 + torch.relu(vecs)) * pooling_mask.unsqueeze(-1), dim=1).values

# Prepare the data
instruction = "Given a web search query, retrieve relevant passages that answer the query."
queries = ["what is rba",
           "what is oilskin fabric"]
instructed_queries = [get_detailed_instruct(instruction, query) for query in queries]
docs = ["Since 2007, the RBA's outstanding reputation has been affected by the 'Securency' or NPA scandal.",
        "Today's oilskins (or oilies) typically come in two parts, jackets and trousers. Oilskin jackets are generally similar to common rubberized waterproofs."]

# Load the model and tokenizer
model = MistralBiForCausalLM.from_pretrained("yibinlei/LENS-d8000", ignore_mismatched_sizes=True)
model.lm_head = torch.load('lm_head.pth')
tokenizer = AutoTokenizer.from_pretrained("yibinlei/LENS-d8000")

# Preprocess the data
query_max_len, doc_max_len = 512, 512
instructed_query_inputs = tokenizer(
                instructed_queries,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=query_max_len,
                add_special_tokens=True
            )
doc_inputs = tokenizer(
                docs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=doc_max_len,
                add_special_tokens=True
            )
# We perform pooling exclusively on the outputs of the query tokens, excluding outputs from the instruction.
query_only_mask = torch.zeros_like(instructed_query_inputs['input_ids'], dtype=instructed_query_inputs['attention_mask'].dtype)
special_token_id = tokenizer.convert_tokens_to_ids('<query>')
for idx, seq in enumerate(instructed_query_inputs['input_ids']):
    special_pos = (seq == special_token_id).nonzero()
    if len(special_pos) > 0:
        query_start_pos = special_pos[-1].item()
        query_only_mask[idx, query_start_pos:-2] = 1 
    else:
        raise ValueError("No special token found")

# Obtain the embeddings
with torch.no_grad():
    instructed_query_outputs = model(**instructed_query_inputs)
    query_embeddings = pooling_func(instructed_query_outputs, query_only_mask)
    doc_outputs = model(**doc_inputs)
    # As the output of each token is used for predicting the next token, the pooling mask is shifted left by 1. The output of the final token EOS token is also excluded.
    doc_inputs['attention_mask'][:, -2:] = 0
    doc_embeddings = pooling_func(doc_outputs, doc_inputs['attention_mask'])

# Normalize the embeddings
query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)

# Compute the similarity
similarity = torch.matmul(query_embeddings, doc_embeddings.T)
```

## Training
We provide detailed training scripts to reproduce our model in the `./finetune` directory. 
Simply run the training with
```
cd ./finetune
bash train.sh
```
You can modify the parameters defined in the `./finetune/train.sh` script to control the training process.

The final model will be saved into the `./finetune/trained_models/` directory.

## Evaluation
We provide the evaluation scripts to evaluate the model on MTEB datasets in the `./eval` directory.
Simply run the evaluation with
```
cd ./eval
python mteb_eval.py --model_name_or_path <path_to_trained_model>
```
The evaluation results will be saved into the `./eval/mteb_results/` directory.
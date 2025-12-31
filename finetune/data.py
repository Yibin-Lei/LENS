import copy
import pickle
import sys

import math
import os.path
import random
from dataclasses import dataclass
from typing import List, Tuple
import json

import numpy as np
import datasets
from numpy import mean
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, DataCollatorForSeq2Seq
from transformers import PreTrainedTokenizer, BatchEncoding
import torch.distributed as dist
import torch

from arguments import DataArguments

def get_query_prompt(query, prompt, use_special_tokens):
    if use_special_tokens:
        return f'<instruct>{prompt}\n<query>{query}'
    else:
        return f'Instruct: {prompt}\nQuery: {query}'


def add_prompt(example, prompt):
    example['prompt'] = prompt
    return example

def traverse_directory_using_os(root_folder):
    file_list = []
    if not os.path.isdir(root_folder):
        file_list.append(root_folder)
    else:
        for dirpath, dirnames, filenames in os.walk(root_folder):
            for filename in filenames:
                full_path = os.path.join(dirpath, filename)
                file_list.append(full_path)
    return file_list


class SameDatasetTrainDataset(Dataset):
    """Dataset to yield a batch of data at one time. All samples in the same batch comes from the same task.
    """

    tokenizer: PreTrainedTokenizer
    loss_type: str

    def __init__(self, args: DataArguments, batch_size, seed, tokenizer, process_index=0, num_processes=1):
        train_datasets = []
        each_data_inxs = []
        batch_size_inxs = []
        data_names = []
        cur_all_num = 0

        FLAG_LOG_NAME = '.log'

        if not args.load_from_disk:
            train_data = args.train_data
            all_dataset = datasets.load_dataset(train_data, cache_dir=args.cache_path)
            for name in all_dataset.keys():
                train_datasets.append(all_dataset[name])
                each_data_inxs.append(np.arange(len(all_dataset[name])) + cur_all_num)
                cur_all_num += len(all_dataset[name])
                if 'symmetric' in all_dataset[name][0]['type']:
                    batch_size_inxs.append(args.symmetric_batch_size // num_processes)
                else:
                    batch_size_inxs.append(batch_size)
                data_names.append(name)

            self.dataset = datasets.concatenate_datasets(train_datasets)
            self.each_data_inxs = each_data_inxs  # k lists, each storing positions of small individual datasets
            self.datasets_inxs = np.arange(len(each_data_inxs))  # k small datasets, indexed from 0 to k-1
            self.batch_size_inxs = batch_size_inxs  # batch size for each small dataset
            self.data_names = data_names

        else:
            # assert isinstance(args.load_disk_path, list) and len(args.load_disk_path) >= 1
            # for load_disk_path in args.load_disk_path:
            load_disk_path = args.load_disk_path
            if not os.path.isdir(load_disk_path):
                raise FileNotFoundError(f"{load_disk_path} is a file, not a directory")
            if not os.path.exists(os.path.join(load_disk_path, FLAG_LOG_NAME)):
                raise FileNotFoundError(f"{load_disk_path} does not have {FLAG_LOG_NAME}")

            with open(os.path.join(load_disk_path, FLAG_LOG_NAME), "r", encoding='utf-8') as f:
                log_info = json.load(f)
                cur_each_data_inxs = [np.array(x) for x in log_info["each_data_inxs"]]
                cur_batch_size_inxs = [batch_size for x in log_info["batch_size_inxs"]]
                cur_data_names = [x for x in log_info["data_names"]]
                print(f"start loading {log_info['train_data']} from {load_disk_path}")
                args.train_data = log_info['train_data']

            cur_dataset = datasets.load_from_disk(load_disk_path)

            for i in range(len(cur_each_data_inxs)):
                cur_each_data_inxs[i] += cur_all_num
            cur_all_num += len(cur_dataset)

            train_datasets.append(cur_dataset)
            each_data_inxs.extend(cur_each_data_inxs)
            batch_size_inxs.extend(cur_batch_size_inxs)
            data_names.extend(cur_data_names)

            self.dataset = datasets.concatenate_datasets(train_datasets)
            self.each_data_inxs = each_data_inxs
            self.datasets_inxs = np.arange(len(each_data_inxs))
            self.batch_size_inxs = batch_size_inxs
            self.data_names = data_names

        if args.save_to_disk:
            if not os.path.exists(args.save_disk_path):
                os.makedirs(args.save_disk_path)
            if os.path.exists(os.path.join(args.save_disk_path, FLAG_LOG_NAME)):
                print(f"FLAG_LOG file {FLAG_LOG_NAME} already exists in {args.save_disk_path}!!!")
                print("args.save_to_disk deprecated.")
            else:
                if args.num_shards <= 0:
                    self.dataset.save_to_disk(args.save_disk_path, max_shard_size=args.save_max_shard_size)
                else:
                    self.dataset.save_to_disk(args.save_disk_path, num_shards=args.num_shards)
                with open(os.path.join(args.save_disk_path, FLAG_LOG_NAME), "w", encoding='utf-8') as f:
                    log_info = {
                        "train_data": args.train_data,
                        "each_data_inxs": [x.tolist() for x in each_data_inxs],
                        "batch_size_inxs": batch_size_inxs,
                        "data_names": data_names
                    }
                    json.dump(log_info, f, ensure_ascii=False, indent=4)
                print(f"save {args.train_data} to {args.save_disk_path}")
            if args.exit_after_save:
                print("exit after save")
                exit(0)

        self.process_index = process_index
        self.num_processes = num_processes
        self.args = args
        self.shuffle_ratio = args.shuffle_ratio

        self.deterministic_generator = np.random.default_rng(seed)
        self.step = 0
        self.refresh_epoch()

        self.tokenizer = tokenizer
        self.query_max_len = self.args.query_max_len
        self.passage_max_len = self.args.passage_max_len

        self.suffix = self.tokenizer('</s>', add_special_tokens=False)['input_ids']
        self.prefix = self.tokenizer('<s>', add_special_tokens=False)['input_ids']

    def refresh_epoch(self):
        print(f'---------------------------*Rank {self.process_index}: refresh data---------------------------')
        self.deterministic_generator.shuffle(self.datasets_inxs)  # shuffle the order of small datasets
        # Dynamically adjust batch size
        batch_datas = []
        for dataset_inx in self.datasets_inxs:  # Load all datasets in the shuffled order of small datasets
            self.deterministic_generator.shuffle(self.each_data_inxs[dataset_inx])  # shuffle the order of data in each dataset
            cur_batch_size = self.batch_size_inxs[
                                 dataset_inx] * self.num_processes  # total batch_size, small batch_size * num_processes
            for start_index in range(0, len(self.each_data_inxs[dataset_inx]), cur_batch_size):
                # judge the last batch's length
                # discard the last incomplete batch
                if start_index + cur_batch_size > len(self.each_data_inxs[dataset_inx]):
                    break
                batch_datas.append(self.each_data_inxs[dataset_inx][start_index:start_index + cur_batch_size])
        self.deterministic_generator.shuffle(batch_datas)  # mix all small datasets
        self.batch_datas = batch_datas
        self.step = 0

    def __getitem__(self, idx):
        if self.step >= len(self.batch_datas):
            self.refresh_epoch()
        batch_indices = self.batch_datas[self.step]
        cur_batch_size = int(len(batch_indices) / self.num_processes)
        batch_indices = batch_indices[self.process_index * cur_batch_size: (
                                               self.process_index + 1) * cur_batch_size]  # only get data for current small batch
        batch_data = self.dataset[batch_indices]
        self.step += 1
        queries_inputs, passages_inputs, messages, scores = self.create_batch_data(batch_raw_data=batch_data)
        return queries_inputs, passages_inputs, messages, scores

    def create_batch_data(self, batch_raw_data):
        queries, passages, scores = [], [], []

        finetune_type = batch_raw_data['type'][0]

        if 'symmetric' in finetune_type and ('sts' in finetune_type or 'clustering' in finetune_type):
            train_group_size = self.args.symmetric_train_group_size
        elif 'only_1neg' in finetune_type:
            train_group_size = 2
        elif 'symmetric' in finetune_type and 'class' in finetune_type:
            train_group_size = self.args.max_class_neg + 1
        else:
            train_group_size = self.args.train_group_size

        for i in range(len(batch_raw_data['query'])):
            queries.append(
                get_query_prompt(batch_raw_data['query'][i], batch_raw_data['prompt'][i], self.args.use_special_tokens))
            pos_index = random.choice(list(range(len(batch_raw_data['pos'][i]))))
            pos = batch_raw_data['pos'][i][pos_index]
            if batch_raw_data.get('pos_scores') is not None:
                if batch_raw_data['pos_scores'][i] is not None:
                    if batch_raw_data['pos_scores'][i][pos_index] is not None:
                        scores.append(batch_raw_data['pos_scores'][i][pos_index])

            if len(batch_raw_data['neg'][i]) < train_group_size - 1:
                num = math.ceil((train_group_size - 1) / len(batch_raw_data['neg'][i]))
                neg_indexes = list(range(len(batch_raw_data['neg'][i]))) * num
            else:
                neg_indexes = list(range(len(batch_raw_data['neg'][i])))
            neg_indexes = random.sample(neg_indexes, train_group_size - 1)
            negs = [batch_raw_data['neg'][i][neg_index] for neg_index in neg_indexes]

            if batch_raw_data.get('neg_scores') is not None:
                try:
                    if batch_raw_data['neg_scores'][i] is not None:
                        for neg_index in neg_indexes:
                            if batch_raw_data['neg_scores'][i][neg_index] is not None:
                                scores.append(batch_raw_data['neg_scores'][i][neg_index])
                except:
                    print(neg_indexes, batch_raw_data['neg_scores'][i])
                    sys.exit()

            tmp_passages = []
            tmp_passages.append(pos)
            tmp_passages.extend(negs)

            # For STS and Clustering, we need to add the instruction to the pos and negs.
            if 'sts' in batch_raw_data['type'][i] or 'clustering' in batch_raw_data['type'][i]:
                tmp_passages = [get_query_prompt(p, batch_raw_data['prompt'][i], self.args.use_special_tokens) for p in
                                tmp_passages]
                tmp_passages = self.tokenizer.batch_decode(
                    self.tokenizer(tmp_passages,
                                   max_length=self.passage_max_len - 1 - len(self.suffix),
                                   truncation=True,
                                   add_special_tokens=False)['input_ids']
                )

            passages.extend(tmp_passages)

        if 'symmetric' in finetune_type and ('class' in finetune_type or 'clustering' in finetune_type):
            messages = ['not in-batch']
        else:
            messages = ['normal'] * len(passages)


        queries_inputs = self.tokenizer(queries,
                                        return_tensors=None,
                                        max_length=self.query_max_len,
                                        truncation=True,
                                        add_special_tokens=True)

        passage_inputs = self.tokenizer(passages,
                                        return_tensors=None,
                                        max_length=self.passage_max_len,
                                        truncation=True,
                                        add_special_tokens=True)

        return queries_inputs, passage_inputs, messages, scores

    def __len__(self):
        return len(self.batch_datas) * self.num_processes


@dataclass
class SameEmbedCollator(DataCollatorForSeq2Seq):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = 32
    passage_max_len: int = 128
    sub_batch_size: int = 0
    train_group_size: int = 0

    def __call__(self, features, return_tensors='pt'):
        if return_tensors is None:
            return_tensors = self.return_tensors

        queries = features[0][0]
        passages = features[0][1]
        messages = features[0][2]
        scores = features[0][3]

        if self.sub_batch_size is None or self.sub_batch_size <= 0:
            raise ValueError("Sub-batch size is not set")
        else:
            batch_size = self.sub_batch_size

            q_collated = []
            for i in range(0, len(queries['attention_mask']), batch_size):
                start = i
                end = min(len(queries['attention_mask']), i + batch_size)
                sub_features = {}
                for k, v in queries.items():
                    sub_features[k] = v[start:end]
                padded = self.tokenizer.pad(
                    sub_features,
                    padding=self.padding,
                    max_length=self.query_max_len,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=return_tensors,
                )
                # Create pooling mask for queries
                pooling_mask = torch.zeros_like(padded['input_ids'], dtype=padded['attention_mask'].dtype)
                special_token_id = self.tokenizer.convert_tokens_to_ids('<query>')
                for idx, seq in enumerate(padded['input_ids']):
                    special_pos = (seq == special_token_id).nonzero()
                    if len(special_pos) > 0:
                        last_pos = special_pos[-1].item()
                        pooling_mask[idx, last_pos:-2] = 1
                    else:
                        pooling_mask[idx, :] = padded['attention_mask'][idx]
                padded['pooling_mask'] = pooling_mask
                q_collated.append(padded)

            d_collated = []
            for i in range(0, len(passages['attention_mask']), batch_size):
                start = i
                end = min(len(passages['attention_mask']), i + batch_size)
                sub_features = {}

                for k, v in passages.items():
                    sub_features[k] = v[start:end]
                padded = self.tokenizer.pad(
                    sub_features,
                    padding=self.padding,
                    max_length=self.passage_max_len,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=return_tensors,
                )
                # Create pooling mask for passages
                pooling_mask = torch.zeros_like(padded['input_ids'], dtype=padded['attention_mask'].dtype)
                special_token_id = self.tokenizer.convert_tokens_to_ids('<query>')
                for idx, seq in enumerate(padded['input_ids']):
                    special_pos = (seq == special_token_id).nonzero()
                    if len(special_pos) > 0:
                        last_pos = special_pos[-1].item()
                        pooling_mask[idx, last_pos:-2] = 1
                    else:
                        pooling_mask[idx, :] = padded['attention_mask'][idx]
                        pooling_mask[:, -2:] = 0  # Set last two positions to 0 because of autoregressive decoding
                padded['pooling_mask'] = pooling_mask
                d_collated.append(padded)

        if len(scores) == 0:
            scores = None

        return {"query": q_collated, "passage": d_collated, 'messages': messages, "teacher_scores": scores}
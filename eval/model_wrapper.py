from mteb.models.text_formatting_utils import corpus_to_texts
from instruction import task_to_instruction
from model import LENSModel
import numpy as np

class LENSWrapper:
    def __init__(self, model_name_or_path=None, max_length=512, batch_size=64, 
                 use_fp16=True, pooling_method=None, bidirectional=True, cache_dir=None):

        self.model = LENSModel(model_name_or_path=model_name_or_path, use_fp16=use_fp16, pooling_method=pooling_method, bidirectional=bidirectional, cache_dir=cache_dir)
        self.max_length = max_length
        self.batch_size = batch_size

    def encode(self, sentences, batch_size=None, *args, prompt_name=None, **kwargs):
        if batch_size is None:
            batch_size = self.batch_size
        else:
            if prompt_name is not None:
                self.model.set_instruction(task_to_instruction(prompt_name))
            return self.model.encode_queries(sentences, batch_size=batch_size, max_length=self.max_length)
    
    def encode_corpus(self, corpus, batch_size=None, *args, prompt_name=None, **kwargs):
        if batch_size is None:
            batch_size = self.batch_size
        corpus = corpus_to_texts(corpus, sep=" ")
        return self.model.encode_corpus(corpus, batch_size=batch_size, max_length=self.max_length)
    
    def encode_queries(self, queries, batch_size=None, *args, prompt_name=None, **kwargs):
        if batch_size is None:
            batch_size = self.batch_size
        if prompt_name is not None:
            self.model.set_instruction(task_to_instruction(prompt_name))
        return self.model.encode_queries(queries, batch_size=batch_size, max_length=self.max_length)

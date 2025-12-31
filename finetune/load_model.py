import os
import re

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from bidirectional_mistral import MistralBiForCausalLM
import numpy as np

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

def get_model(model_args, output_dir, resize, resize_tokens):

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name,
                                            token=model_args.token,
                                            cache_dir=model_args.cache_dir,
                                            )
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path,
                                            token=model_args.token,
                                            cache_dir=model_args.cache_dir,
                                            )
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )
    config.use_cache = False

    if model_args.model_name_or_path:
        if model_args.bidirectional:
            model = MistralBiForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                use_flash_attention_2=True if model_args.use_flash_attn else False,
                token=model_args.token,
                cache_dir=model_args.cache_dir,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                use_flash_attention_2=True if model_args.use_flash_attn else False,
                token=model_args.token,
                cache_dir=model_args.cache_dir,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
            )
    else:
        print("Training new model from scratch")
        model = model_args.from_config(config)

    if model_args.raw_peft is not None:
        model.set_input_embeddings(torch.load(os.path.join(model_args.raw_peft, 'embedding', 'emb.pth')))
        model = PeftModel.from_pretrained(model, model_args.raw_peft)
        model = model.merge_and_unload()

    if resize:
        model.resize_token_embeddings(resize_tokens)
        os.makedirs(os.path.join(output_dir, 'embedding'), exist_ok=True)
        torch.save(model.get_input_embeddings(), os.path.join(output_dir, 'embedding', 'input_emb.pth'))
        target_modules = model_args.target_modules
    else:
        target_modules = model_args.target_modules
        if 'embed_tokens' in target_modules:
            target_modules.remove('embed_tokens')

    # Load the new lm head
    if model_args.lm_head_path is not None:
        new_lm_head = np.load(model_args.lm_head_path)
        # Create a new linear layer with the desired shape
        import torch.nn as nn
        new_lm_head_tensor = torch.tensor(new_lm_head, dtype=model.lm_head.weight.dtype, device=model.lm_head.weight.device)
        new_linear_head = nn.Linear(model.lm_head.in_features, new_lm_head_tensor.size(0), bias=model.lm_head.bias is not None)
        new_linear_head.weight.data = new_lm_head_tensor
        if model.lm_head.bias is not None:
            raise ValueError("Bias is not supported for the new lm_head")
        
        # Replace the old lm_head with the new one
        model.lm_head = new_linear_head
        torch.save(model.lm_head, os.path.join(output_dir, 'embedding', 'lm_head.pth'))

    if model_args.from_peft is not None:
        if os.path.exists(os.path.join(model_args.from_peft, 'embedding')):
            model.set_input_embeddings(torch.load(os.path.join(model_args.from_peft, 'embedding', 'emb.pth')))
            torch.save(model.embed_tokens, os.path.join(output_dir, 'embedding', 'emb.pth'))
        model = PeftModel.from_pretrained(model, model_args.from_peft, is_trainable=True)
        model.print_trainable_parameters()
    else:
        if model_args.use_lora:
            peft_config = LoraConfig(
                inference_mode=False,
                r=model_args.lora_rank,
                target_modules=target_modules,
                modules_to_save=model_args.modules_to_save,
                lora_alpha=model_args.lora_alpha,
                lora_dropout=model_args.lora_dropout
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
    print(model)
    return model

def save_merged_model(model_args, output_dir):
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name,
                                            token=model_args.token,
                                            cache_dir=model_args.cache_dir,
                                            )
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path,
                                            token=model_args.token,
                                            cache_dir=model_args.cache_dir,
                                            )
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )
    config.use_cache = False

    if model_args.model_name_or_path:
        model = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            use_flash_attention_2=True if model_args.use_flash_attn else False,
            token=model_args.token,
            cache_dir=model_args.cache_dir,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
        )
    else:
        print("Training new model from scratch")
        model = model_args.from_config(config)

    if model_args.raw_peft is not None:
        model.set_input_embeddings(torch.load(os.path.join(model_args.raw_peft, 'embedding', 'emb.pth')))
        model = PeftModel.from_pretrained(model, model_args.raw_peft)
        model = model.merge_and_unload()

    if os.path.exists(os.path.join(output_dir, 'embedding', 'emb.pth')):
        model.set_input_embeddings(torch.load(os.path.join(output_dir, 'embedding', 'emb.pth')))
    if os.path.exists(os.path.join(output_dir, 'embedding', 'lm_head.pth')):
        model.lm_head = torch.load(os.path.join(output_dir, 'embedding', 'lm_head.pth'))
        
    try:
        model = PeftModel.from_pretrained(model, output_dir)
        model = model.merge_and_unload()
    except:
        model = PeftModel.from_pretrained(model, find_largest_checkpoint(output_dir))
        model = model.merge_and_unload()

    model.save_pretrained(os.path.join(output_dir, 'full_model'))

    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    tokenizer.save_pretrained(os.path.join(output_dir, 'full_model'))
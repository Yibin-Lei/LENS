POOLING_METHOD=max-pooling
LM_HEAD_PATH=./lm_head/4000.npy
BIDIRECTIONAL=True
TRAIN_BATCH_SIZE=32
CACHE_PATH=<path_to_cache_dir>
TOKEN=<huggingface_token>

LM_HEAD_NAME=$(basename $LM_HEAD_PATH)
RUN_NAME=${POOLING_METHOD}-bi${BIDIRECTIONAL}-${LM_HEAD_NAME%.*}token

# Train
torchrun \
--nnodes 4 \
--nproc_per_node 4 \
run.py \
--output_dir ./trained_models/${RUN_NAME} \
--model_name_or_path mistralai/Mistral-7B-v0.1 \
--train_data cfli/bge-full-data \
--learning_rate 1e-4 \
--num_train_epochs 1 \
--per_device_train_batch_size $TRAIN_BATCH_SIZE \
--lora_alpha 64 \
--lora_rank 32 \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 512 \
--passage_max_len 512 \
--train_group_size 8 \
--logging_steps 1 \
--save_steps 500 \
--save_total_limit 10000 \
--ddp_find_unused_parameters False \
--negatives_cross_device \
--gradient_checkpointing \
--deepspeed ./deepspeed_stage1.json \
--warmup_steps 100 \
--fp16 True \
--cache_dir $CACHE_PATH \
--token $TOKEN \
--cache_path $CACHE_PATH \
--sub_batch_size 64 \
--target_modules q_proj k_proj v_proj o_proj down_proj up_proj gate_proj \
--use_special_tokens \
--symmetric_batch_size 256 \
--symmetric_train_group_size 8 \
--max_class_neg 7 \
--save_merged_lora_model True \
--pooling_method $POOLING_METHOD \
--lm_head_path $LM_HEAD_PATH \
--bidirectional $BIDIRECTIONAL \
--run_name $RUN_NAME
# 
export MODEL_NAME="/data/wan2.1basecode/VideoX-Fun/models/Wan2.1-Fun-V1.1-14B-InP"
# export MODEL_NAME="/data/wan2.1basecode/VideoX-Fun/models/Wan2.1-Fun-V1.1-1.3B-InP"
export DATASET_NAME= None
export DATASET_META_NAME="/data/wan2.1basecode/VideoX-Fun/datasets/internal_datasets/large_output.json"
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

#export NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4
#export NCCL_NET=IB
export NCCL_SOCKET_IFNAME=eth1

RANK=$MLP_ROLE_INDEX
MASTER_ADDR=$MLP_WORKER_0_HOST
MASTER_PORT=$MLP_WORKER_0_PORT

accelerate launch --num_machines 2 --num_processes 16 --machine_rank ${RANK} --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT} \
  --zero_stage 3 --zero3_save_16bit_model true --zero3_init_flag true --use_deepspeed --deepspeed_config_file config/zero_stage3_config.json --deepspeed_multinode_launcher standard\
  scripts/wan2.1_fun/train_stage_1_1.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=960 \
  --video_sample_size=256 \
  --token_sample_size=960 \
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=16 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="/nas/accumulation16" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --train_mode="inpaint" \
  --trainable_modules "."
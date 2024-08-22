#!/bin/bash

# requirements.txt in project dir will be automatically installed by SageMaker
# pip install -r requirements.txt


# ## 在训练节点上直接下载
# rm -rf LLaMA-Factory
# git clone https://github.com/hiyouga/LLaMA-Factory.git


cat > "/tmp/data-path/dataset_info.json" << EOF
{
  "customized_data": {
    "file_name": "alpaca_zh_demo.json"
  }
}
EOF


# Predefined torchrun config
DISTRIBUTED_ARGS="--nproc_per_node $SM_NUM_GPUS --nnodes $TORCHRUN_NODE_NUMBER --node_rank $TORCHRUN_NODE_INDEX --master_addr $TORCHRUN_MASTER --master_port 7777"

# --dataset_dir /opt/ml/code/LLaMA-Factory/data \
# --dataset alpaca_gpt4_zh \

torchrun $DISTRIBUTED_ARGS /opt/ml/code/LLaMA-Factory/src/train.py \
    --deepspeed /opt/ml/code/ds-config-z3-offload-act.json \
    --stage sft \
    --do_train True \
    --model_name_or_path /tmp/initial-model-path \
    --finetuning_type full \
    --template qwen \
    --dataset_dir /tmp/data-path/ \
    --dataset customized_data \
    --cutoff_len 512 \
    --learning_rate 2e-6 \
    --num_train_epochs 1 \
    --max_steps 10 \
    --max_samples 100000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --preprocessing_num_workers 4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 50 \
    --save_steps 100 \
    --warmup_steps 0 \
    --optim adamw_torch \
    --report_to none \
    --output_dir /tmp/tuned-model-path \
    --bf16 True








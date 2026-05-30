#!/bin/bash

# 设置目标显存空闲阈值（60GB，单位MB）
TARGET_FREE_MEM=61440

# 检查 nvidia-smi 是否可用
if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi not found. Please ensure NVIDIA drivers are installed."
    exit 1
fi

dataset="All_Beauty"
output_dir="./SASRec/${dataset}/Seq100/id(GFT)+text_all_fft/seed_42"
mode="train"  # 默认值
gpu_specified="1"

# 创建 output_dir 目录（如果不存在）
mkdir -p "$output_dir"

# 解析参数：支持 --mode 和 --gpu
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --mode)
            mode="$2"
            shift
            ;;
        --gpu)
            gpu_specified="$2"
            shift
            ;;
        *)
            ;;
    esac
    shift
done

# 设置日志文件
if [[ "$mode" == "train" ]]; then
    log_file="${output_dir}/training.txt"
elif [[ "$mode" == "test" ]]; then
    log_file="${output_dir}/test_new.txt"
else
    echo "Unknown mode: $mode. Using default log file name."
    log_file="${output_dir}/output_log.txt"
fi

# 函数：检查可用GPU（在自动模式下使用）
check_gpu() {
    local gpu_info
    gpu_info=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)

    while IFS=, read -r gpu_id free_mem; do
        gpu_id=$(echo "$gpu_id" | tr -d ' ')
        free_mem=$(echo "$free_mem" | tr -d ' ')

        echo "GPU $gpu_id: Free memory = ${free_mem}MB"

        if [ "$free_mem" -ge "$TARGET_FREE_MEM" ]; then
            echo "Found suitable GPU $gpu_id with ${free_mem}MB free memory!"
            return "$gpu_id"
        fi
    done <<< "$gpu_info"

    return 255
}

# 函数：运行主任务
run_task() {
    python main.py \
        --seed 42 \
        --mode "$mode" \
        --num_check 9 \
        --rec_model_path /data/mhwang/LLM/Fre_LLM/Pre_Train_Rec_Model/sasrec/checkpoint/All_Beauty/SASRec.epoch=200.lr=0.001.layer=2.head=1.hidden=50.maxlen=100.pth \
        --checkpoint "${output_dir}/trained_params_final.pt" \
        --train_path ./dataset/process_data/${dataset}/Train_data.df \
        --val_path ./dataset/process_data/${dataset}/Val_data.df \
        --test_path ./dataset/process_data/${dataset}/Test_data.df \
        --id2name_path ./dataset/process_data/${dataset}/id2name.txt \
        --pretrained_model /data/mhwang/LLM-Research/Qwen2.5-7B-Instruct  \
        --max_seq_len 100 \
        --num_negative_samples 99 \
        --batch_size 32 \
        --epochs 15 \
        --lr 5e-4 \
        --output_dir "$output_dir" \
        --accumulate_grad_batches 4 \
        --lr_decay_min_lr 5e-6 \
        --lr_warmup_start_lr 5e-6 \
        --proj_intermediate_dim 1000 \
        --embed_mode None \
        --num_trainable_items 0 \
        --hidden_layer -1 \
        --init_fa 0.1 \
        --lower_quantile 0.0 \
        --upper_quantile 0.75 \
        --inject_every_layer 
}
hold_gpu() {
    echo "Task completed. Now holding GPU memory..."
    python -c "import torch; a = torch.ones(1024*1024*50, device='cuda'); print('Holding 50GB of GPU memory'); while True: pass" 2>/dev/null
}

# 主逻辑
if [[ -n "$gpu_specified" ]]; then
    echo "Using manually specified GPU: $gpu_specified"
    export CUDA_VISIBLE_DEVICES=$gpu_specified

    # 将标准输出和标准错误输出重定向到文件
    exec > >(tee "$log_file") 2>&1

    # 运行任务
    run_task

    # 任务完成后保持GPU占用
    hold_gpu
else
    echo "Monitoring GPUs for ${TARGET_FREE_MEM}MB free memory..."
    while true; do
        check_gpu
        selected_gpu=$?
        if [ "$selected_gpu" != 255 ]; then
            echo "Assigning task to GPU $selected_gp"
            export CUDA_VISIBLE_DEVICES=$selected_gpu

            # 将标准输出和标准错误输出重定向到文件
            exec > >(tee "$log_file") 2>&1

            # 运行任务
            run_task

            # 任务完成后保持GPU占用
            hold_gpu

            # 由于hold_gpu包含无限循环，这里实际上不会到达
            break
        else
            echo "No GPU with sufficient memory found. Waiting..."
            sleep 300
        fi
    done
fi
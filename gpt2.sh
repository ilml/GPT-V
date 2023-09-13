python3 --num_gpus=1  trainer.py \
    --model_type gpt2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --num_train_epochs 1 \
    --output_dir ./tmp \
    --overwrite_output_dir true\
    --fp16 true 
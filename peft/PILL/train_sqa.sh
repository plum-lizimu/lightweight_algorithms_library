# torchrun --nproc_per_node=1 /data/zfy/stich_adapter/train_sqa.py \
#                              --output_dir=/data/zfy/stich_adapter/output_dir/full \
#                              --log_dir=/data/zfy/stich_adapter/output_dir/full \
python attn_gate.py
python attn_gate_analyze.py

# torchrun --nproc_per_node=1 --master_port 11111 train_sqa.py \
#                             --output_dir=./output_dir/repeat_4 \
#                             --log_dir=./output_dir/repeat_4 \
#                             --seed=42 \

# python eval_sqa.py --adapter_path=./output_dir/repeat_4/checkpoint-19.pth

# screen -ls
# screen -r 3380319.pts-9.LTserver
# ctrl a + atrl d
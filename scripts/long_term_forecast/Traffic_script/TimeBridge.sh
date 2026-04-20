# export CUDA_VISIBLE_DEVICES=0,1

model_name=TimeBridge

seq_len=96
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.0002
d_model=512
d_ff=512
train_epochs=100
patience=10
batch_size=64

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  /scratch/s223669184/project_data/Grant25/TimeSeriesECM/dataset/dataset/ \
  --data_path traffic.csv \
  --model_id Traffic_$seq_len'_'96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 96 \
  --e_layers $e_layers \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size 128 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --ca_layers 3 \
  --pd_layers 1 \
  --ia_layers 1 \
  --alpha 0.35 \
  --num_p 12 \
  --period 48 \
  --use_multi_gpu \
  --devices '0,1' \

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /scratch/s223669184/project_data/Grant25/TimeSeriesECM/dataset/dataset/\
  --data_path traffic.csv \
  --model_id Traffic_$seq_len'_'192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 192 \
  --e_layers $e_layers \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size 128 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --ca_layers 3 \
  --pd_layers 1 \
  --ia_layers 1 \
  --alpha 0.35 \
  --num_p 12 \
  --period 48 \
  --use_multi_gpu \
  --devices '0,1' \

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /scratch/s223669184/project_data/Grant25/TimeSeriesECM/dataset/dataset/\
  --data_path traffic.csv \
  --model_id Traffic_$seq_len'_'336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 336 \
  --e_layers $e_layers \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size 128 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --ca_layers 3 \
  --pd_layers 1 \
  --ia_layers 1 \
  --alpha 0.35 \
  --num_p 12 \
  --period 48 \
  --use_multi_gpu \
  --devices '0,1' \

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /scratch/s223669184/project_data/Grant25/TimeSeriesECM/dataset/dataset/\
  --data_path traffic.csv \
  --model_id Traffic_$seq_len'_'720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 720 \
  --e_layers $e_layers \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size 128 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --ca_layers 3 \
  --pd_layers 1 \
  --ia_layers 1 \
  --alpha 0.35 \
  --num_p 12 \
  --period 48 \
  --use_multi_gpu \
  --devices '0,1' \

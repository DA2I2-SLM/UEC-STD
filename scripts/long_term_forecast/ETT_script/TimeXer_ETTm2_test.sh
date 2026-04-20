# export CUDA_VISIBLE_DEVICES=1

model_name=TimeXer

seq_len=96
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
train_epochs=10
patience=10
batch_size=16
kernel_size=25

python -u run.py \
  --task_name long_term_forecast \
  --is_training $1 \
  --root_path /scratch/s223669184/project_data/Grant25/TimeSeriesECM/dataset/dataset/\
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'$3 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $3 \
  --e_layers $e_layers \
  --enc_in 7 \
  --c_out 7 \
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
  --use_ar $2 \
  --errcor_coef $4 \
  --err_h 32 \
  --ecm_model $5\
  --season_coef $6 \
  --trend_coef $7 \
  --kernel_size $kernel_size \


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#    --root_path  /scratch/s223669184/project_data/Grant25/TimeSeriesECM/dataset/dataset/ \#   --data_path ETTm2.csv \
#   --model_id ETTm2_$seq_len'_'192 \
#   --model $model_name \
#   --data ETTm2 \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 192 \
#   --e_layers $e_layers \
#   --enc_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --learning_rate $learning_rate \
#   --train_epochs $train_epochs \
#   --patience $patience \
#   --batch_size 128 \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#    --root_path  /scratch/s223669184/project_data/Grant25/TimeSeriesECM/dataset/dataset/ \#   --data_path ETTm2.csv \
#   --model_id ETTm2_$seq_len'_'336 \
#   --model $model_name \
#   --data ETTm2 \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 336 \
#   --e_layers $e_layers \
#   --enc_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --learning_rate $learning_rate \
#   --train_epochs $train_epochs \
#   --patience $patience \
#   --batch_size 128 \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#    --root_path  /scratch/s223669184/project_data/Grant25/TimeSeriesECM/dataset/dataset/ \#   --data_path ETTm2.csv \
#   --model_id ETTm2_$seq_len'_'720 \
#   --model $model_name \
#   --data ETTm2 \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 0 \
#   --pred_len 720 \
#   --e_layers $e_layers \
#   --enc_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --learning_rate $learning_rate \
#   --train_epochs $train_epochs \
#   --patience $patience \
#   --batch_size 128 \
#   --down_sampling_layers $down_sampling_layers \
#   --down_sampling_method avg \
#   --down_sampling_window $down_sampling_window

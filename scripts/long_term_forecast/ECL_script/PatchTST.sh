#export CUDA_VISIBLE_DEVICES=0

model_name=PatchTST

seq_len=96
e_layers=2
d_layers=1
label_len=48
factor=3
train_epochs=10
patience=3
batch_size=16
learning_rate=0.0001
n_heads=4
kernel_size=25

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /scratch/s223669184/project_data/Grant25/TimeSeriesECM/dataset/dataset/ \
  --data_path electricity.csv \
  --model_id ECL_$seq_len'_'96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len 96 \
  --e_layers $e_layers \
  --d_layers $d_layers \
  --factor $factor \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --n_heads $n_heads \
  --itr 1 \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --kernel_size $kernel_size

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /scratch/s223669184/project_data/Grant25/TimeSeriesECM/dataset/dataset/ \
  --data_path electricity.csv \
  --model_id ECL_$seq_len'_'192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len 192 \
  --e_layers $e_layers \
  --d_layers $d_layers \
  --factor $factor \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --n_heads $n_heads \
  --itr 1 \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --kernel_size $kernel_size

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /scratch/s223669184/project_data/Grant25/TimeSeriesECM/dataset/dataset/ \
  --data_path electricity.csv \
  --model_id ECL_$seq_len'_'336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len 336 \
  --e_layers $e_layers \
  --d_layers $d_layers \
  --factor $factor \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --n_heads $n_heads \
  --itr 1 \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --kernel_size $kernel_size

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /scratch/s223669184/project_data/Grant25/TimeSeriesECM/dataset/dataset/ \
  --data_path electricity.csv \
  --model_id ECL_$seq_len'_'720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len 720 \
  --e_layers $e_layers \
  --d_layers $d_layers \
  --factor $factor \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --n_heads $n_heads \
  --itr 1 \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --kernel_size $kernel_size

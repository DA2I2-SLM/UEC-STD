#export CUDA_VISIBLE_DEVICES=0

model_name=PatchTST

seq_len=96
e_layers=2
d_layers=1
label_len=48
factor=3
train_epochs=10
patience=3
batch_size=4
learning_rate=0.0001
n_heads=4
kernel_size=25

python -u run.py \
  --task_name long_term_forecast \
  --is_training $1 \
  --root_path /scratch/s223669184/project_data/Grant25/TimeSeriesECM/dataset/dataset/ \
  --data_path traffic.csv \
  --model_id Traffic_$seq_len'_'$3 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $3 \
  --e_layers $e_layers \
  --d_layers $d_layers \
  --factor $factor \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --itr 1 \
  --n_heads $n_heads \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size $batch_size \
  --use_ar $2 \
  --errcor_coef $4 \
  --err_h 32 \
  --ecm_model $5 \
  --season_coef $6 \
  --trend_coef $7 \
  --kernel_size $kernel_size

#export CUDA_VISIBLE_DEVICES=0

model_name=PatchTST

seq_len=96
e_layers=2
n_heads=4
label_len=48
train_epochs=10
patience=10
kernel_size=25

python -u run.py \
  --task_name long_term_forecast \
  --is_training $1 \
  --root_path /scratch/s223669184/project_data/Grant25/TimeSeriesECM/dataset/dataset/ \
  --data_path weather.csv \
  --model_id weather_96_$3 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $3 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --n_heads $n_heads \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size 128 \
  --use_ar $2 \
  --errcor_coef $4 \
  --err_h 32 \
  --ecm_model $5 \
  --season_coef $6 \
  --trend_coef $7 \
  --kernel_size $kernel_size \

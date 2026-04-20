model_name=TimesNet

seq_len=96
e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
train_epochs=20
patience=10
kernel_size=25

for pred_len in 96 192 336 720; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 9 \
    --root_path /scratch/s223669184/project_data/Grant25/TimeSeriesECM/dataset/dataset/ \
    --data_path weather.csv \
    --model_id weather_${seq_len}_${pred_len} \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers $e_layers \
    --d_layers 1 \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --itr 1 \
    --d_model $d_model \
    --d_ff $d_ff \
    --batch_size 128 \
    --learning_rate $learning_rate \
    --train_epochs $train_epochs \
    --patience $patience \
    --down_sampling_layers $down_sampling_layers \
    --down_sampling_method avg \
    --down_sampling_window $down_sampling_window \
    --use_ar 0 \
    --errcor_coef 0.5 \
    --err_h 32 \
    --ecm_model linear \
    --season_coef 0.6 \
    --trend_coef 0.4 \
    --kernel_size $kernel_size \
    --error_flags "0.1,0.3,0.5,0.7,1.0"
done

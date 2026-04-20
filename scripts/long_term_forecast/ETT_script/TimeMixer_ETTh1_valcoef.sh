model_name=TimeMixer

seq_len=96
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
train_epochs=10
patience=10
kernel_size=25

for pred_len in 96 192 336 720; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 9 \
    --root_path /scratch/s223669184/project_data/Grant25/TimeSeriesECM/dataset/dataset/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_${seq_len}_${pred_len} \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
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
    --use_ar 0 \
    --errcor_coef 0.5 \
    --err_h 32 \
    --ecm_model linear \
    --season_coef 0.6 \
    --trend_coef 0.4 \
    --kernel_size $kernel_size \
    --error_flags "0.1,0.3,0.5,0.7,1.0"
done

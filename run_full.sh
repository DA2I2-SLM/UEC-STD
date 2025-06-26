#!/bin/bash
#SBATCH --job-name=run_full_job_ecm
#SBATCH --output=run_full_job_ecm_%j.out
#SBATCH --error=run_full_job_ecm_%j.err
#SBATCH --time=96:00:00         # Adjust as needed
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5        # Adjust CPU cores if needed
#SBATCH --mem=80G                # Adjust memory if needed
#SBATCH --gres=gpu:1             # Request 1 GPU if needed, remove if not
#SBATCH --qos=batch-long


# Load modules if needed, e.g., conda, python, etc.
module load Anaconda3
source activate
conda activate timeecm

# Run your command
export CUDA_VISIBLE_DEVICES=""
# Train the initial model 

# python run_full.py --data ETTh1 --data_type ETT --model TimeMixer --ecm linear,logistic,random_forest,xgboost,lstm,GRU,CNN,TF --step 1 
# python run_full.py --data Traffic,Weather --data_type Traffic,Weather --model TimeMixer --ecm linear,logistic,random_forest,xgboost,lstm,GRU,CNN,TF --step 1

# Train the ECMs for step 2 and step 3 without seasonal and trend components

# python run_full.py --data ETTh1 --data_type ETT --model TimeMixer --ecm linear,logistic,random_forest,xgboost,lstm,GRU,CNN,TF --step 2 --seasonal_trend False --season_coef 0.5 --trend_coef 0.5 &
# python run_full.py --data ETTh1 --data_type ETT --model TimeMixer --ecm linear,logistic,random_forest,xgboost,lstm,GRU,CNN,TF --step 3 --seasonal_trend False --season_coef 0.5 --trend_coef 0.5 &

# python run_full.py --data Traffic --data_type Traffic --model TimeMixer --ecm linear,logistic,random_forest,xgboost,lstm,GRU,CNN,TF --step 2 --seasonal_trend False --season_coef 0.5 --trend_coef 0.5 &
# python run_full.py --data Traffic --data_type Traffic --model TimeMixer --ecm linear,logistic,random_forest,xgboost,lstm,GRU,CNN,TF --step 3 --seasonal_trend False --season_coef 0.5 --trend_coef 0.5 &

# python run_full.py --data Weather --data_type Weather --model TimeMixer --ecm linear,logistic,random_forest,xgboost,lstm,GRU,CNN,TF --step 2 --seasonal_trend False --season_coef 0.5 --trend_coef 0.5 &
# python run_full.py --data Weather --data_type Weather --model TimeMixer --ecm linear,logistic,random_forest,xgboost,lstm,GRU,CNN,TF --step 3 --seasonal_trend False --season_coef 0.5 --trend_coef 0.5 &


# Train the ECMs for step 2 and step 3 with seasonal and trend components

# python run_full.py --data ETTh1 --data_type ETT --model TimeMixer --ecm linear,logistic,random_forest,xgboost,lstm,GRU,CNN,TF --step 2 --seasonal_trend True --season_coef 0.5 --trend_coef 0.5 &
# python run_full.py --data ETTh1 --data_type ETT --model TimeMixer --ecm linear,logistic,random_forest,xgboost,lstm,GRU,CNN,TF --step 3 --seasonal_trend True --season_coef 0.5 --trend_coef 0.5

# python run_full.py --data Traffic --data_type Traffic --model TimeMixer --ecm linear,logistic,random_forest,xgboost,lstm,GRU,CNN,TF --step 2 --seasonal_trend True --season_coef 0.5 --trend_coef 0.5 &
# python run_full.py --data Traffic --data_type Traffic --model TimeMixer --ecm linear,logistic,random_forest,xgboost,lstm,GRU,CNN,TF --step 3 --seasonal_trend True --season_coef 0.5 --trend_coef 0.5

# python run_full.py --data Weather --data_type Weather --model TimeMixer --ecm linear,logistic,random_forest,xgboost,lstm,GRU,CNN,TF --step 2 --seasonal_trend True --season_coef 0.5 --trend_coef 0.5 &
# python run_full.py --data Weather --data_type Weather --model TimeMixer --ecm linear,logistic,random_forest,xgboost,lstm,GRU,CNN,TF --step 3 --seasonal_trend True --season_coef 0.5 --trend_coef 0.5


# Change the seasonal and trend weights

python run_full.py --data ETTh1 --data_type ETT --model TimeMixer --ecm linear,logistic,random_forest,xgboost,lstm,GRU,CNN,TF --step 3 --seasonal_trend True --season_coef 0.4 --trend_coef 0.6

python run_full.py --data Traffic --data_type Traffic --model TimeMixer --ecm linear,logistic,random_forest,xgboost,lstm,GRU,CNN,TF --step 3 --seasonal_trend True --season_coef 0.4 --trend_coef 0.6

python run_full.py --data Weather --data_type Weather --model TimeMixer --ecm linear,logistic,random_forest,xgboost,lstm,GRU,CNN,TF --step 3 --seasonal_trend True --season_coef 0.4 --trend_coef 0.6



python run_full.py --data ETTh1 --data_type ETT --model TimeMixer --ecm linear,logistic,random_forest,xgboost,lstm,GRU,CNN,TF --step 3 --seasonal_trend True --season_coef 0.3 --trend_coef 0.7

python run_full.py --data Traffic --data_type Traffic --model TimeMixer --ecm linear,logistic,random_forest,xgboost,lstm,GRU,CNN,TF --step 3 --seasonal_trend True --season_coef 0.3 --trend_coef 0.7

python run_full.py --data Weather --data_type Weather --model TimeMixer --ecm linear,logistic,random_forest,xgboost,lstm,GRU,CNN,TF --step 3 --seasonal_trend True --season_coef 0.3 --trend_coef 0.7

wait

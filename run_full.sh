#!/bin/bash
#SBATCH --job-name=run_full_job_ecm
#SBATCH --output=run_full_job_ecm_%j.out
#SBATCH --error=run_full_job_ecm_%j.err
#SBATCH --time=96:00:00         # Adjust as needed
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2        # Adjust CPU cores if needed
#SBATCH --mem=32G                # Adjust memory if needed
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

# python run_full.py --data ETTh1 --data_type ETT --model TimeMixer --ecm linear,logistic,random_forest,xgboost,lstm,GRU,CNN,TF --step 2 --seasonal_trend False --season_coef 0.5 --trend_coef 0.5
python run_full.py --data ETTh1 --data_type ETT --model TimeMixer --ecm linear --step 3 --seasonal_trend False --season_coef 0.5 --trend_coef 0.5

python run_full.py --data Traffic,Weather --data_type Traffic,Weather --model TimeMixer --ecm linear --step 2 --seasonal_trend False --season_coef 0.5 --trend_coef 0.5
python run_full.py --data Traffic,Weather --data_type Traffic,Weather --model TimeMixer --ecm linear --step 3 --seasonal_trend False --season_coef 0.5 --trend_coef 0.5


# Train the ECMs for step 2 and step 3 with seasonal and trend components

# python run_full.py --data ETTh1 --data_type ETT --model TimeMixer --ecm linear,logistic,random_forest,xgboost,lstm,GRU,CNN,TF --step 2 --seasonal_trend True --season_coef 0.5 --trend_coef 0.5
# python run_full.py --data ETTh1 --data_type ETT --model TimeMixer --ecm linear,logistic,random_forest,xgboost,lstm,GRU,CNN,TF --step 3 --seasonal_trend True --season_coef 0.5 --trend_coef 0.5

python run_full.py --data Traffic,Weather --data_type Traffic,Weather --model TimeMixer --ecm linear --step 2 --seasonal_trend True --season_coef 0.5 --trend_coef 0.5
python run_full.py --data Traffic,Weather --data_type Traffic,Weather --model TimeMixer --ecm linear --step 3 --seasonal_trend True --season_coef 0.5 --trend_coef 0.5

# Change the seasonal and trend weights

# python run_full.py --data ETTh1 --data_type ETT --model TimeMixer --ecm linear,logistic,random_forest,xgboost,lstm,GRU,CNN,TF --step 2 --seasonal_trend True --season_coef 0.6 --trend_coef 0.4
# python run_full.py --data ETTh1 --data_type ETT --model TimeMixer --ecm linear,logistic,random_forest,xgboost,lstm,GRU,CNN,TF --step 3 --seasonal_trend True --season_coef 0.6 --trend_coef 0.4

# python run_full.py --data Traffic,Weather --data_type Traffic,Weather --model TimeMixer --ecm linear,logistic,random_forest,xgboost,lstm,GRU,CNN,TF --step 2 --seasonal_trend True --season_coef 0.6 --trend_coef 0.4
# python run_full.py --data Traffic,Weather --data_type Traffic,Weather --model TimeMixer --ecm linear,logistic,random_forest,xgboost,lstm,GRU,CNN,TF --step 3 --seasonal_trend True --season_coef 0.6 --trend_coef 0.4



# python run_full.py --data ETTh1 --data_type ETT --model TimeMixer --ecm linear,logistic,random_forest,xgboost,lstm,GRU,CNN,TF --step 2 --seasonal_trend True --season_coef 0.7 --trend_coef 0.3
# python run_full.py --data ETTh1 --data_type ETT --model TimeMixer --ecm linear,logistic,random_forest,xgboost,lstm,GRU,CNN,TF --step 3 --seasonal_trend True --season_coef 0.7 --trend_coef 0.3

# python run_full.py --data Traffic,Weather --data_type Traffic,Weather --model TimeMixer --ecm linear,logistic,random_forest,xgboost,lstm,GRU,CNN,TF --step 2 --seasonal_trend True --season_coef 0.7 --trend_coef 0.3
# python run_full.py --data Traffic,Weather --data_type Traffic,Weather --model TimeMixer --ecm linear,logistic,random_forest,xgboost,lstm,GRU,CNN,TF --step 3 --seasonal_trend True --season_coef 0.7 --trend_coef 0.3





# python run_full.py --data Traffic,Weather --data_type Traffic,Weather --model TimeMixer,iTransformer,Mamba,TSMixer --ecm linear,logistic,xgboost,lstm,CNN --step 2
# python run_full.py --data Traffic,ETTh1,Weather --data_type Traffic,ETT,Weather --model TimeMixer,iTransformer,Mamba,TSMixer --ecm linear,logistic,xgboost,lstm,CNN --step 3
# python run_full.py --data ETTh1 --data_type ETT --model TimeMixer,iTransformer,Mamba,TSMixer --ecm linear,logistic,xgboost,lstm,CNN --step 3

# python run_eval.py DummyFileName --data_files=VASP_LGPS_ChemMater_2018_30_4995_MD_repeat_800K.OUTCAR.RAWDELTA,VASP_LGPS_ChemMater_2018_30_4995_MD_repeat_1000K.OUTCAR.RAWDELTA,VASP_LGPS_ChemMater_2018_30_4995_MD_repeat_1200K.OUTCAR.RAWDELTA --models=TimeMixer,iTransformer,Mamba,TSMixer --eng_coefs=0,0.001,0.0005,0.0001 --eng_samples=100 --start_run_id=54 --train=0

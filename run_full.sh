#!/bin/bash
#SBATCH --job-name=timexer
#SBATCH --output=timexer.out
#SBATCH --error=timexer.err
#SBATCH --time=96:00:00         # Adjust as needed
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10        # Adjust CPU cores if needed
#SBATCH --mem=64G                # Adjust memory if needed
#SBATCH --gres=gpu:1             # Request 1 GPU if needed, remove if not
#SBATCH --qos=batch-long


# Load modules if needed, e.g., conda, python, etc.
module load Anaconda3
source activate
conda activate timeecm

# Run your command
# export CUDA_VISIBLE_DEVICES=""
# Train the initial model 

# python run_full.py --data custom --data_type ECL --model TimeMixer,TimesNet,TimeXer --ecm linear,logistic,random_forest,xgboost,lstm,GRU,CNN,TF --step 1 

# Running without seasonal and trend coefficients

# python run_full.py --data custom --data_type ECL --model TimeMixer,TimesNet,TimeXer --ecm logistic --step 2 --ablation 0
# python run_full.py --data custom --data_type ECL --model TimeMixer,TimesNet,TimeXer --ecm logistic --step 3 --ablation 0

# python run_full.py --data custom --data_type ECL --model TimesNet,TimeXer --ecm lstm,GRU,CNN,TF,linear --step 2 --ablation 0
# python run_full.py --data custom --data_type ECL --model TimesNet,TimeXer --ecm lstm,GRU,CNN,TF,linear --step 3 --ablation 0


# python run_full.py --data custom --data_type ECL --model TimesNet,TimeXer --ecm logistic,random_forest,xgboost --step 2 --ablation 0
# python run_full.py --data custom --data_type ECL --model TimesNet,TimeXer --ecm logistic,random_forest,xgboost --step 3 --ablation 0

# python run_full.py --data custom --data_type ECL --model TimeMixer --ecm logistic,random_forest,xgboost --step 2 --ablation 0
# python run_full.py --data custom --data_type ECL --model TimeMixer --ecm logistic,random_forest,xgboost --step 3 --ablation 0


# python run_full.py --data custom --data_type Exchange --model TimesNet,TimeMixer,TimeXer --ecm lstm,GRU,CNN,TF,linear --step 2 --ablation 0
# python run_full.py --data custom --data_type Exchange --model TimesNet,TimeMixer,TimeXer --ecm lstm,GRU,CNN,TF,linear --step 3 --ablation 0

# python run_full.py --data custom --data_type Exchange --model TimeMixer,TimeXer,TimesNet --ecm logistic,random_forest,xgboost --step 2 --ablation 0
# python run_full.py --data custom --data_type Exchange --model TimeMixer,TimeXer,TimesNet --ecm logistic,random_forest,xgboost --step 3 --ablation 0


python run_full.py --data custom --data_type ECL --model TimeXer --ecm linear,lstm,GRU,CNN,TF --step 2
python run_full.py --data custom --data_type ECL --model TimeXer --ecm linear,lstm,GRU,CNN,TF --step 3

python run_full.py --data custom --data_type ECL --model TimeXer --ecm logistic,random_forest,xgboost --step 2
python run_full.py --data custom --data_type ECL --model TimeXer --ecm logistic,random_forest,xgboost --step 3


# python run_full.py --data Weather --data_type Weather --model TimeMixer,TimeXer,TimesNet --ecm linear,lstm,GRU,CNN,TF --step 2
# python run_full.py --data Weather --data_type Weather --model TimeMixer,TimeXer,TimesNet --ecm linear,lstm,GRU,CNN,TF --step 3

# python run_full.py --data Weather --data_type Weather --model TimeMixer,TimeXer,TimesNet --ecm logistic,random_forest,xgboost --step 2
# python run_full.py --data Weather --data_type Weather --model TimeMixer,TimeXer,TimesNet --ecm logistic,random_forest,xgboost --step 3

# python run_full.py --data ETTh2 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm linear,lstm,GRU,CNN,TF --step 2
# python run_full.py --data ETTh2 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm linear,lstm,GRU,CNN,TF --step 3

# python run_full.py --data ETTh2 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm logistic,random_forest,xgboost --step 2
# python run_full.py --data ETTh2 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm logistic,random_forest,xgboost --step 3



# python run_full.py --data ETTm1 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm linear,lstm,GRU,CNN,TF --step 2
# python run_full.py --data ETTm1 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm linear,lstm,GRU,CNN,TF --step 3

# python run_full.py --data ETTm1 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm logistic,random_forest,xgboost --step 2
# python run_full.py --data ETTm1 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm logistic,random_forest,xgboost --step 3

# python run_full.py --data ETTm2 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm linear,lstm,GRU,CNN,TF --step 2
# python run_full.py --data ETTm2 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm linear,lstm,GRU,CNN,TF --step 3

# python run_full.py --data ETTm2 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm logistic,random_forest,xgboost --step 2
# python run_full.py --data ETTm2 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm logistic,random_forest,xgboost --step 3









# Running with seasonal and trend coefficients


# python run_full.py --data custom --data_type ECL --model TimesNet,TimeMixer,TimeXer --ecm linear --seasonal_trend True --season_coef 0.4 --trend_coef 0.6 --step 2 --ablation 0
# python run_full.py --data custom --data_type ECL --model TimesNet,TimeMixer,TimeXer --ecm linear --seasonal_trend True --season_coef 0.4 --trend_coef 0.6 --step 3 --ablation 0

# python run_full.py --data custom --data_type ECL --model TimesNet,TimeMixer,TimeXer --ecm linear --seasonal_trend True --season_coef 0.6 --trend_coef 0.4 --step 2 --ablation 0
# python run_full.py --data custom --data_type ECL --model TimesNet,TimeMixer,TimeXer --ecm linear --seasonal_trend True --season_coef 0.6 --trend_coef 0.4 --step 3 --ablation 0

# python run_full.py --data custom --data_type ECL --model TimesNet,TimeMixer,TimeXer --ecm linear --seasonal_trend True --season_coef 0.7 --trend_coef 0.3 --step 2 --ablation 0
# python run_full.py --data custom --data_type ECL --model TimesNet,TimeMixer,TimeXer --ecm linear --seasonal_trend True --season_coef 0.7 --trend_coef 0.3 --step 3 --ablation 0

# python run_full.py --data custom --data_type ECL --model TimesNet,TimeMixer,TimeXer --ecm linear --seasonal_trend True --season_coef 0.8 --trend_coef 0.2 --step 2 --ablation 0
# python run_full.py --data custom --data_type ECL --model TimesNet,TimeMixer,TimeXer --ecm linear --seasonal_trend True --season_coef 0.8 --trend_coef 0.2 --step 3 --ablation 0


# python run_full.py --data ETTh1 --data_type ETT --model TimeXer --ecm linear --step 2 --seasonal_trend True --season_coef 0.8 --trend_coef 0.2
# python run_full.py --data ETTh1 --data_type ETT --model TimeXer --ecm linear --step 3 --seasonal_trend True --season_coef 0.8 --trend_coef 0.2

# python run_full.py --data Traffic --data_type Traffic --model TimeMixer --ecm linear  --step 2 --seasonal_trend True --season_coef 0.0 --trend_coef 1.0 --ablation 1
# python run_full.py --data Traffic --data_type Traffic --model TimeMixer --ecm linear  --step 3 --seasonal_trend True --season_coef 0.0 --trend_coef 1.0 --ablation 1

# python run_full.py --data Traffic --data_type Traffic --model TimeMixer --ecm linear  --step 2 --seasonal_trend True --season_coef 0.0 --trend_coef 1.0 --ablation 0
# python run_full.py --data Traffic --data_type Traffic --model TimeMixer --ecm linear  --step 3 --seasonal_trend True --season_coef 0.0 --trend_coef 1.0 --ablation 0

# python run_full.py --data Traffic --data_type Traffic --model TimeMixer --ecm linear  --step 2 --seasonal_trend True --season_coef 1.0 --trend_coef 0.0 --ablation 0
# python run_full.py --data Traffic --data_type Traffic --model TimeMixer --ecm linear  --step 3 --seasonal_trend True --season_coef 1.0 --trend_coef 0.0 --ablation 0

# python run_full.py --data Weather --data_type Weather --model TimeMixer --ecm linear  --step 2 --seasonal_trend True --season_coef 0.0 --trend_coef 1.0 --ablation 1
# python run_full.py --data Weather --data_type Weather --model TimeMixer --ecm linear  --step 3 --seasonal_trend True --season_coef 0.0 --trend_coef 1.0 --ablation 1

# python run_full.py --data Weather --data_type Weather --model TimeMixer --ecm linear  --step 2 --seasonal_trend True --season_coef 0.0 --trend_coef 1.0 --ablation 0
# python run_full.py --data Weather --data_type Weather --model TimeMixer --ecm linear  --step 3 --seasonal_trend True --season_coef 0.0 --trend_coef 1.0 --ablation 0

# python run_full.py --data Weather --data_type Weather --model TimeMixer --ecm linear  --step 2 --seasonal_trend True --season_coef 1.0 --trend_coef 0.0 --ablation 0
# python run_full.py --data Weather --data_type Weather --model TimeMixer --ecm linear  --step 3 --seasonal_trend True --season_coef 1.0 --trend_coef 0.0 --ablation 0



# python run_full.py --data ETTh1 --data_type ETT --model TimeMixer --ecm linear  --step 2 --seasonal_trend True --season_coef 1.0 --trend_coef 0.0
# python run_full.py --data ETTh1 --data_type ETT --model TimeMixer --ecm linear  --step 3 --seasonal_trend True --season_coef 1.0 --trend_coef 0.0


# python run_full.py --data ETTh2 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm linear  --step 2 --seasonal_trend True --season_coef 0.0 --trend_coef 1.0
# python run_full.py --data ETTh2 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm linear  --step 3 --seasonal_trend True --season_coef 0.0 --trend_coef 1.0

# python run_full.py --data ETTh2 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm linear  --step 2 --seasonal_trend True --season_coef 1.0 --trend_coef 0.0
# python run_full.py --data ETTh2 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm linear  --step 3 --seasonal_trend True --season_coef 1.0 --trend_coef 0.0

# python run_full.py --data ETTh1 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm linear  --step 2 --seasonal_trend True --season_coef 0.2 --trend_coef 0.8
# python run_full.py --data ETTh1 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm linear  --step 3 --seasonal_trend True --season_coef 0.2 --trend_coef 0.8




# python run_full.py --data ETTh2 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm linear --step 2 --seasonal_trend True --season_coef 0.8 --trend_coef 0.2
# python run_full.py --data ETTh2 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm linear --step 3 --seasonal_trend True --season_coef 0.8 --trend_coef 0.2

# python run_full.py --data ETTh2 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm linear  --step 2 --seasonal_trend True --season_coef 0.4 --trend_coef 0.6
# python run_full.py --data ETTh2 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm linear  --step 3 --seasonal_trend True --season_coef 0.4 --trend_coef 0.6

# python run_full.py --data ETTh2 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm linear  --step 2 --seasonal_trend True --season_coef 0.2 --trend_coef 0.8
# python run_full.py --data ETTh2 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm linear  --step 3 --seasonal_trend True --season_coef 0.2 --trend_coef 0.8

# python run_full.py --data Traffic --data_type Traffic --model TimeMixer,TimeXer,TimesNet --ecm linear --step 2 --seasonal_trend True --season_coef 0.8 --trend_coef 0.2
# python run_full.py --data Traffic --data_type Traffic --model TimeMixer,TimeXer,TimesNet --ecm linear --step 3 --seasonal_trend True --season_coef 0.8 --trend_coef 0.2

# python run_full.py --data Traffic --data_type Traffic --model TimeMixer,TimeXer,TimesNet --ecm linear  --step 2 --seasonal_trend True --season_coef 0.4 --trend_coef 0.6
# python run_full.py --data Traffic --data_type Traffic --model TimeMixer,TimeXer,TimesNet --ecm linear  --step 3 --seasonal_trend True --season_coef 0.4 --trend_coef 0.6

# python run_full.py --data Traffic --data_type Traffic --model TimeMixer,TimeXer,TimesNet --ecm linear  --step 2 --seasonal_trend True --season_coef 0.2 --trend_coef 0.8
# python run_full.py --data Traffic --data_type Traffic --model TimeMixer,TimeXer,TimesNet --ecm linear  --step 3 --seasonal_trend True --season_coef 0.2 --trend_coef 0.8



# python run_full.py --data Weather --data_type Weather --model TimeMixer,TimeXer,TimesNet --ecm linear  --step 2 --seasonal_trend True --season_coef 0.8 --trend_coef 0.2
# python run_full.py --data Weather --data_type Weather --model TimeMixer,TimeXer,TimesNet --ecm linear  --step 3 --seasonal_trend True --season_coef 0.8 --trend_coef 0.2

# python run_full.py --data Weather --data_type Weather --model TimeMixer,TimeXer,TimesNet --ecm linear  --step 2 --seasonal_trend True --season_coef 0.4 --trend_coef 0.6
# python run_full.py --data Weather --data_type Weather --model TimeMixer,TimeXer,TimesNet --ecm linear  --step 3 --seasonal_trend True --season_coef 0.4 --trend_coef 0.6

# python run_full.py --data Weather --data_type Weather --model TimeMixer,TimeXer,TimesNet --ecm linear  --step 2 --seasonal_trend True --season_coef 0.2 --trend_coef 0.8
# python run_full.py --data Weather --data_type Weather --model TimeMixer,TimeXer,TimesNet --ecm linear  --step 3 --seasonal_trend True --season_coef 0.2 --trend_coef 0.8


# python run_full.py --data ETTm1 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm linear --step 2 --seasonal_trend True --season_coef 0.8 --trend_coef 0.2
# python run_full.py --data ETTm1 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm linear --step 3 --seasonal_trend True --season_coef 0.8 --trend_coef 0.2

# python run_full.py --data ETTm1 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm linear  --step 2 --seasonal_trend True --season_coef 0.4 --trend_coef 0.6
# python run_full.py --data ETTm1 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm linear  --step 3 --seasonal_trend True --season_coef 0.4 --trend_coef 0.6

# python run_full.py --data ETTm1 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm linear  --step 2 --seasonal_trend True --season_coef 0.2 --trend_coef 0.8
# python run_full.py --data ETTm1 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm linear  --step 3 --seasonal_trend True --season_coef 0.2 --trend_coef 0.8


# python run_full.py --data ETTm2 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm linear --step 2 --seasonal_trend True --season_coef 0.8 --trend_coef 0.2
# python run_full.py --data ETTm2 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm linear --step 3 --seasonal_trend True --season_coef 0.8 --trend_coef 0.2

# python run_full.py --data ETTm2 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm linear  --step 2 --seasonal_trend True --season_coef 0.4 --trend_coef 0.6
# python run_full.py --data ETTm2 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm linear  --step 3 --seasonal_trend True --season_coef 0.4 --trend_coef 0.6

# python run_full.py --data ETTm2 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm linear  --step 2 --seasonal_trend True --season_coef 0.2 --trend_coef 0.8
# python run_full.py --data ETTm2 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm linear  --step 3 --seasonal_trend True --season_coef 0.2 --trend_coef 0.8







# python run_full.py --data ETTm1 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm TF  --step 2 --seasonal_trend True --season_coef 0.8 --trend_coef 0.2
# python run_full.py --data ETTm1 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm TF  --step 3 --seasonal_trend True --season_coef 0.8 --trend_coef 0.2

# python run_full.py --data ETTm1 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm TF  --step 2 --seasonal_trend True --season_coef 0.4 --trend_coef 0.6
# python run_full.py --data ETTm1 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm TF  --step 3 --seasonal_trend True --season_coef 0.4 --trend_coef 0.6

# python run_full.py --data ETTm1 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm TF  --step 2 --seasonal_trend True --season_coef 0.2 --trend_coef 0.8
# python run_full.py --data ETTm1 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm TF  --step 3 --seasonal_trend True --season_coef 0.2 --trend_coef 0.8


# python run_full.py --data ETTm2 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm TF  --step 2 --seasonal_trend True --season_coef 0.8 --trend_coef 0.2
# python run_full.py --data ETTm2 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm TF  --step 3 --seasonal_trend True --season_coef 0.8 --trend_coef 0.2

# python run_full.py --data ETTm2 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm TF  --step 2 --seasonal_trend True --season_coef 0.4 --trend_coef 0.6
# python run_full.py --data ETTm2 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm TF  --step 3 --seasonal_trend True --season_coef 0.4 --trend_coef 0.6

# python run_full.py --data ETTm2 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm TF  --step 2 --seasonal_trend True --season_coef 0.2 --trend_coef 0.8
# python run_full.py --data ETTm2 --data_type ETT --model TimeMixer,TimeXer,TimesNet --ecm TF  --step 3 --seasonal_trend True --season_coef 0.2 --trend_coef 0.8



# python run_full.py --data Traffic --data_type Traffic --model TimeMixer,TimeXer,TimesNet --ecm TF  --step 2 --seasonal_trend True --season_coef 0.8 --trend_coef 0.2
# python run_full.py --data Traffic --data_type Traffic --model TimeMixer,TimeXer,TimesNet --ecm TF  --step 3 --seasonal_trend True --season_coef 0.8 --trend_coef 0.2

# python run_full.py --data Traffic --data_type Traffic --model TimeMixer,TimeXer,TimesNet --ecm TF  --step 2 --seasonal_trend True --season_coef 0.4 --trend_coef 0.6
# python run_full.py --data Traffic --data_type Traffic --model TimeMixer,TimeXer,TimesNet --ecm TF  --step 3 --seasonal_trend True --season_coef 0.4 --trend_coef 0.6

# python run_full.py --data Traffic --data_type Traffic --model TimeMixer,TimeXer,TimesNet --ecm TF  --step 2 --seasonal_trend True --season_coef 0.2 --trend_coef 0.8
# python run_full.py --data Traffic --data_type Traffic --model TimeMixer,TimeXer,TimesNet --ecm TF  --step 3 --seasonal_trend True --season_coef 0.2 --trend_coef 0.8


# python run_full.py --data Weather --data_type Weather --model TimeMixer,TimeXer,TimesNet --ecm TF  --step 2 --seasonal_trend True --season_coef 0.8 --trend_coef 0.2
# python run_full.py --data Weather --data_type Weather --model TimeMixer,TimeXer,TimesNet --ecm TF  --step 3 --seasonal_trend True --season_coef 0.8 --trend_coef 0.2

# python run_full.py --data Weather --data_type Weather --model TimeMixer,TimeXer,TimesNet --ecm TF  --step 2 --seasonal_trend True --season_coef 0.4 --trend_coef 0.6
# python run_full.py --data Weather --data_type Weather --model TimeMixer,TimeXer,TimesNet --ecm TF  --step 3 --seasonal_trend True --season_coef 0.4 --trend_coef 0.6

# python run_full.py --data Weather --data_type Weather --model TimeMixer,TimeXer,TimesNet --ecm TF  --step 2 --seasonal_trend True --season_coef 0.2 --trend_coef 0.8
# python run_full.py --data Weather --data_type Weather --model TimeMixer,TimeXer,TimesNet --ecm TF  --step 3 --seasonal_trend True --season_coef 0.2 --trend_coef 0.8





wait

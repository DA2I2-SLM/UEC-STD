# TimeSeriesECM
Timeseries With Error Correction Model (Autoregressive Decoding)

Reference Repo: https://github.com/thuml/Time-Series-Library

## Setup
```bash
# Install Python
conda create -n timeecm python=3.8
conda activate timeecm
# Install other dependencies
pip install -r requirements.txt
```

## Dataset Preparation
```bash
cd TimeSeriesECM
mkdir scratch
mkdir scratch/dataset/
```
Download and unzip the dataset https://github.com/thuml/Time-Series-Library

## Experiment Steps
Follow these steps to create and evaluate ECM
#### Train the backbone model
For example, train TimeMixer with ETTh1 dataset:
```bash
bash ./scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1.sh
```
The checkpoints and logs should be found in ./scratch/s223669184/project_data/Grant25/TimeSeriesECM/checkpoints and ./scratch/results

#### Train the ECM model
First, create the run script for ECM. The run script is very similar to the backbone training script (e.g., TimeMixer_ETTh1.sh). 
For example, the ECM script for TimeMixer and Etth1 is TimeMixer_ETTh1_test.sh. Then, run the following:
```bash
bash ./scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1_test.sh 3 1 384 1 "linear"
bash ./scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1_test.sh 3 1 384 1 "logistic"
bash ./scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1_test.sh 3 1 384 1 "random_forest"
bash ./scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1_test.sh 3 1 336 1 "xgboost"
bash ./scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1_test.sh 3 1 336 1 "lstm"
bash ./scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1_test.sh 3 1 336 1 "GRU"
bash ./scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1_test.sh 3 1 336 1 "CNN"
bash ./scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1_test.sh 3 1 336 1 "TF" "True"

```
/scratch/s223669184/project_data/Grant25/TimeSeriesECM/dataset/checkpoints/long_term_forecast_ETTh1_96_96_TimeMixer_ETTh1_ftM_sl96_ll0_pl96_dm16_nh8_el2_dl1_df32_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0

Argument explanation:
- 3: A special mode of doing inference that generates data for training ECM and conducts the training
- 1: Turn on autoregressive decoding. It should be always 1. If it is set to 0, the input for autoregressive decoding will be the ground truth.
- 384: The length of the decoding sequence. It should be multiples of the unit prediction length (eg., 96)
- 1: Turn on the rate of error-correcting. It is not used in mode 3

The ECM model is saved in the same checkpoint directory as the backbone model.
 
#### Evaluate the ECM model
Run the ECM script with different arguments
```bash
bash ./scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1_test.sh 4 1 336 1 "linear"
bash ./scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1_test.sh 4 1 336 1 "logistic"
bash ./scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1_test.sh 4 1 336 1 "random_forest"
bash ./scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1_test.sh 4 1 336 1 "xgboost"
bash ./scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1_test.sh 4 1 336 1 "lstm"
bash ./scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1_test.sh 4 1 336 1 "GRU"
bash ./scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1_test.sh 4 1 336 1 "CNN"
bash ./scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1_test.sh 4 1 336 1 "TF"


```
Argument explanation:
- 4: A special mode of doing inference that evaluates the ECM
- 1: Turn on autoregressive decoding (must be 1).
- 336: the testing length (336) can be different from the training length (384).
- 1: It can be any number between 0 and 1. 0 means not using ECM, and 1 means fully using ECM. The number in between specifies how much ECM is used.
  
To do a mass evaluation with different decoding lengths and error-correcting rates:
```python
python run_eval.py --data ETTh1 --model TimeMixer --ecm "linear"
python run_eval.py --data ETTh1 --model TimeMixer --ecm "logistic"
python run_eval.py --data ETTh1 --model TimeMixer --ecm "random_forest"
python run_eval.py --data ETTh1 --model TimeMixer --ecm "xgboost"
python run_eval.py --data ETTh1 --model TimeMixer --ecm "lstm"
python run_eval.py --data ETTh1 --model TimeMixer --ecm "GRU"
python run_eval.py --data ETTh1 --model TimeMixer --ecm "CNN"
python run_eval.py --data ETTh1 --model TimeMixer --ecm "TF" --include_x0 "True"


```
The results should be found in ./scratch/infer_results/

#### Report the ECM evaluation
Report the TimeMixer ECM on ETTh1 dataset
```python
python report.py --dir ./scratch/infer_results/long_term_forecast_ETTh1_96_96_TimeMixer_ETTh1_ftM_sl96_ll0_pl96_dm16_nh8_el2_dl1_df32_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0/
```

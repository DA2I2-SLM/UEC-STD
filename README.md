# TimeSeriesECM
Timeseries With Error Corrrection Model (Autoregressive Decoding)

Reference Repo: https://github.com/thuml/Time-Series-Library

## Setup
```bash
# Install Python
conda create -n timemd python=3.8
conda activate timeecm
# Install other dependencies
pip install -r requirements.txt
```

## Experiment Steps
Follow these steps to create and evaluate ECM
#### Train the backbone model
For example, train TimeMixer with ETTh1 dataset:
```bash
bash ./scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1.sh
```
The checkpoints and logs should be found in ./scratch/

#### Train the ECM model
First, create the run script for ECM. The run script is very similar to the backbone training script (e.g., TimeMixer_ETTh1.sh). 
For example, the ECM script for TimeMixer and Etth1 is TimeMixer_ETTh1_test.sh. Then, run the following:
```bash
bash ./scripts/long_term_forecast/ETTh1_script/TimeMixer_ETTh1_test.sh 3 1 384 1
```
Argument explaination:
- 3: A special mode of doing inference that generate data for training ECM and conduct the training
- 1: Turn on autoregressive decoding. It should be always 1. If it set to 0, the input for autoregressive decoding will be the ground truth.
- 384: The length of the decoding sequence. It should be multiples of the unit prediction length (96)
- 1: Turn on the rate of error correcting. It is not used in mode 3
 
#### Evaluate the ECM model
Run the ECM script with different arguments
```bash
bash ./scripts/long_term_forecast/ETTh1_script/TimeMixer_ETTh1_test.sh 4 1 336 1
```
- 4: A special mode of doing inference that evaluate the ECM
- Note: the testing length (336) can be different from the training length (384).
To do mass evaluation with different decoding lengths and error correcting rates:
```python
python run_eval.py --data ETTh1 --model TimeMixer
```



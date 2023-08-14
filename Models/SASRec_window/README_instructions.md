# README

## Data

The data used in this project is stored in the `data` folder and consists of two datasets:
- `ml-1m.txt`: The main dataset for model training and evaluation.
- `ml-1m_time.txt`: An extended dataset that includes temporal information, specifically used with Time2Vec.

## Code Organization

The `SASRec_window` directory comprises various scripts and notebooks essential for the experiments:

1. **main.py**: The main script to run, train, and evaluate the model.
2. **model.py**: Contains the model's architecture, forward, and prediction methods. This also includes the SASRec model with Time2Vec temporal embeddings.
3. **utils.py**: Houses utility functions such as evaluation methods, data partitioning methods, and sampling methods.
4. **graphs.py**: A script used to generate graphs for the third experiment. (Note: Results are manually hard-coded, so it won't generate all window graphs by default.)
5. **MovieLens_1M_Analysis.ipynb**: Jupyter notebook used for Exploratory Data Analysis (EDA) on the MovieLens 1M dataset.

## Experiments

### 1. First Experiment: Performance of Training Objectives
Commands to run the first experiment:

```bash
# Baseline Next Item Prediction
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=bce --model_training=None --window_eval=true --uniform_ss=false --window_size=7  # With BCE
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=sampled_softmax --model_training=None --window_eval=true --uniform_ss=true --window_size=7  # With SS-Uniform 
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=sampled_softmax --model_training=None --window_eval=true --uniform_ss=false --window_size=7  # With SS-LogQ correction

# Skip Item Prediction
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=skip --loss_type=bce --model_training=None --window_eval=true --uniform_ss=false --window_size=7   # With BCE
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=skip --loss_type=sampled_softmax --model_training=None --window_eval=true --uniform_ss=true --window_size=7    # With SS-Uniform 
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=skip --loss_type=sampled_softmax --model_training=None --window_eval=true --uniform_ss=false --window_size=7    # With LogQ correction (same order with losses in the following commands)

# Incremental Item Prediction 
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=incremental --loss_type=bce --model_training=None --window_eval=true --uniform_ss=false --window_size=7
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=incremental --loss_type=sampled_softmax --model_training=None --window_eval=true --uniform_ss=true --window_size=7
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=incremental --loss_type=sampled_softmax --model_training=None --window_eval=true --uniform_ss=false --window_size=7

# All Action Prediction 
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=ce_over --model_training=all_action --window_eval=true --uniform_ss=false --window_size=7
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=sampled_softmax --model_training=all_action --window_eval=true --uniform_ss=true --window_size=7
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=sampled_softmax --model_training=all_action --window_eval=true --uniform_ss=false --window_size=7

# Dense All Action Prediction
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=ce_over --model_training=dense_all_action --window_eval=true --uniform_ss=false --window_size=7
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=sampled_softmax --model_training=dense_all_action --window_eval=true --uniform_ss=true --window_size=7
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=sampled_softmax --model_training=dense_all_action --window_eval=true --uniform_ss=false --window_size=7

# Super Dense All Action Prediction
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=ce_over --model_training=super_dense_all_action --window_eval=true --uniform_ss=false --window_size=7
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=sampled_softmax --model_training=super_dense_all_action --window_eval=true --uniform_ss=true --window_size=7
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=sampled_softmax --model_training=super_dense_all_action --window_eval=true --uniform_ss=false --window_size=7

# Rolling Future Window Prediction
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=ce_over --model_training=future_rolling --window_eval=true --uniform_ss=false --window_size=7
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=sampled_softmax --model_training=future_rolling --window_eval=true --uniform_ss=true --window_size=7
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=sampled_softmax --model_training=future_rolling --window_eval=true --uniform_ss=false --window_size=7

# Integrated All Action Prediction
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --model_training=combined --loss_type=ce_over --window_eval=true --window_size=7
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --model_training=combined --loss_type=sampled_softmax --uniform_ss=true --window_eval=true --window_size=7
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --model_training=combined --loss_type=sampled_softmax --uniform_ss=false --window_eval=true --window_size=7
```

### 2. Second Experiment: Effects of Training Techniques and Temporal Information
Commands to run the second experiment:

```bash
# All following with Integrated All Action Prediction
# Time2Vec
python main.py --dataset=ml-1m_time --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --loss_type=ce_over --model_training=combined --window_eval=true --uniform_ss=false --window_size=7  --temporal=true
python main.py --dataset=ml-1m_time --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --loss_type=sampled_softmax --model_training=combined --window_eval=true --uniform_ss=true --window_size=7 --temporal=true

# Auto-regressive technique 
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --loss_type=ce_over --model_training=combined --window_eval=true --uniform_ss=false --strategy=autoregressive --window_size=7
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --loss_type=sampled_softmax --model_training=combined --window_eval=true --uniform_ss=true --strategy=autoregressive --window_size=7

# Teacher Forcing technique
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --loss_type=ce_over --model_training=combined --window_eval=true --uniform_ss=false --strategy=teacher_forcing --window_size=7
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --loss_type=sampled_softmax --model_training=combined --window_eval=true --uniform_ss=true --strategy=teacher_forcing --window_size=7

# Masking technique (15%)
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=ce_over --model_training=combined --window_eval=true --uniform_ss=false --masking=true --mask_prob=0.15
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=sampled_softmax --model_training=combined --window_eval=true --uniform_ss=true --masking=true --mask_prob=0.15

# Masking (15%) + Teacher Forcing techniques 
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=ce_over --model_training=combined --strategy=teacher_forcing --window_eval=true --uniform_ss=false --masking=true
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=sampled_softmax --model_training=combined --strategy=teacher_forcing --window_eval=true --uniform_ss=true --masking=true
```

### 3. Third Experiment: Impact of Different Window Sizes
Commands to run the third experiment:

```bash
# All following with Integrated All Action Prediction using SS-U loss
# Window 3
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --model_training=combined --loss_type=sampled_softmax --uniform_ss=true --window_eval=true --window_size=3 --window_eval_size=3
# Window 10
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --model_training=combined --loss_type=sampled_softmax --uniform_ss=true --window_eval=true --window_size=10 --window_eval_size=10
# Window 14
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --model_training=combined --loss_type=sampled_softmax --uniform_ss=true --window_eval=true --window_size=14 --window_eval_size=14
```

## Notes
- For a detailed understanding of the arguments passed in the commands, refer to the respective Python files.
- The experiments aim to address specific research questions and are designed to build upon the findings of the previous ones.
- Evaluation metrics include Hit Rate@10, NDCG@10, and Kendall's Tau (as detailed in Section `3.4 Evaluation Process`).
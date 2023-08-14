# MSc AI Thesis, University of Edinburgh, 2023
## Title: Recommender Systems: Looking further into the future 
## Author: Kyriakos Kyriakou
## Abstract: 
Recommender systems play a crucial role in various sectors, from entertainment apps to e-commerce platforms, by offering personalized recommendations based on users' preferences and historical interactions. However, many of these systems primarily focus on predicting the immediate next action of users, potentially missing out on complex patterns in users' behaviors. This thesis presents an in-depth exploration of sequential recommendation systems, with the primary goal of improving the prediction of future user interactions. The research expands upon the work of Pancha et al. (2022), offering a divergent approach by focusing on the sequential order of future interactions, rather than their explicit temporal context. Using the SASRec model as a foundation, we introduce several training objectives and techniques, and evaluate their effectiveness on the widely-used MovieLens-1M dataset.
Our findings reveal that our Integrated All Action Prediction modeling approach, combined with the Sampled Softmax Uniform loss, outperforms other methods in predicting future user interactions. This hybrid approach integrates the strengths of both Next Item Prediction and long-term All Action Prediction, achieving superior performance across several key metrics, including NDCG@10, Hit@10, and Kendall's Tau.
The study also investigates the impact of various training techniques and the size of future prediction windows. Among these, Teacher Forcing particularly stands out for its proficiency in predicting a well-ordered sequence of future items. Additionally, extending the future prediction window size indicates potential long-term benefits in recommendation accuracy.
Despite the significant contributions of this research, it opens new avenues for future work. These findings offer valuable insights and pave the way for the development of more effective and robust sequential recommendation systems.

## Repository Structure:
- Dataset Analysis - EDA on dataset
- IPP - Informatics Project Proposal Report
- Models - Models used. SASRec_window main folder for future predictions-proposed model.
- Dissertation Report

## Window-Based Recommender System - SASRec_window folder README:
### Data

The data used in this project is stored in the `data` folder and consists of two datasets:
- `ml-1m.txt`: The main dataset for model training and evaluation.
- `ml-1m_time.txt`: An extended dataset that includes temporal information, specifically used with Time2Vec.

### Code Organization

The `SASRec_window` directory comprises various scripts and notebooks essential for the experiments:

1. **main.py**: The main script to run, train, and evaluate the model.
2. **model.py**: Contains the model's architecture, forward, and prediction methods. This also includes the SASRec model with Time2Vec temporal embeddings.
3. **utils.py**: Houses utility functions such as evaluation methods, data partitioning methods, and sampling methods.
4. **graphs.py**: A script used to generate graphs for the third experiment. (Note: Results are manually hard-coded, so it won't generate all window graphs by default.)
5. **MovieLens_1M_Analysis.ipynb**: Jupyter notebook used for Exploratory Data Analysis (EDA) on the MovieLens 1M dataset.

### Experiments

#### 1. First Experiment: Performance of Training Objectives
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

#### 2. Second Experiment: Effects of Training Techniques and Temporal Information
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

#### 3. Third Experiment: Impact of Different Window Sizes
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

### Notes
- For a detailed understanding of the arguments passed in the commands, refer to the respective Python files.
- The experiments aim to address specific research questions and are designed to build upon the findings of the previous ones.
- Evaluation metrics include Hit Rate@10, NDCG@10, and Kendall's Tau (as detailed in Section `3.4 Evaluation Process`).

### References
Model modified based on [paper author's tensorflow implementation](https://github.com/kang205/SASRec), switching to PyTorch(v1.6) for simplicity.
Model base code (before all changes) adopted from: [SASRec-PyTorch repo](https://github.com/pmixer/SASRec.pytorch)
Check paper author's [repo](https://github.com/kang205/SASRec) for detailed intro and more complete README, and here's paper bib:

```
@inproceedings{kang2018self,
  title={Self-attentive sequential recommendation},
  author={Kang, Wang-Cheng and McAuley, Julian},
  booktitle={2018 IEEE International Conference on Data Mining (ICDM)},
  pages={197--206},
  year={2018},
  organization={IEEE}
}
```

For Time2Vec implementation check paper: `Time2Vec: Learning a Vector representation of Time`.
Time2Vec base code adopted from: [Time2Vec-PyTorch repo](https://github.com/ojus1/Time2Vec-PyTorch). Here is paper bib:

```
@misc{kazemi2019time2vec,
  title={Time2Vec: Learning a Vector Representation of Time}, 
  author={Seyed Mehran Kazemi and Rishab Goel and Sepehr Eghbali and Janahan Ramanan and Jaspreet Sahota and Sanjay Thakur and Stella Wu and Cathal Smyth and Pascal Poupart and Marcus Brubaker},
  year={2019},
  eprint={1907.05321},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

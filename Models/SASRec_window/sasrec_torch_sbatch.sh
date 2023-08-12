#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-08:00:00

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}


export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

mkdir -p ${TMP}/datasets/
export DATASET_DIR=${TMP}/datasets/
# Activate the relevant virtual environment:


#source /home/${STUDENT_ID}/miniconda3/bin/activate mscproject

# First Exp:
#base
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=bce --model_training=None --window_eval=true --uniform_ss=false --window_size=7
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=sampled_softmax --model_training=None --window_eval=true --uniform_ss=true --window_size=7
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=sampled_softmax --model_training=None --window_eval=true --uniform_ss=false --window_size=7

#skip
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=independent --loss_type=bce --model_training=None --window_eval=true --uniform_ss=false --window_size=7
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=independent --loss_type=sampled_softmax --model_training=None --window_eval=true --uniform_ss=true --window_size=7

#incremental 
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=teacher_forcing --loss_type=bce --model_training=None --window_eval=true --uniform_ss=false --window_size=7
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=teacher_forcing --loss_type=sampled_softmax --model_training=None --window_eval=true --uniform_ss=true --window_size=7

#all action 
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=ce_over --model_training=all_action --window_eval=true --uniform_ss=false --window_size=7
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=sampled_softmax --model_training=all_action --window_eval=true --uniform_ss=true --window_size=7
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=sampled_softmax --model_training=all_action --window_eval=true --uniform_ss=false --window_size=7

#dense all action 
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=ce_over --model_training=dense_all_action --window_eval=true --uniform_ss=false --window_size=7
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=sampled_softmax --model_training=dense_all_action --window_eval=true --uniform_ss=true --window_size=7

#super dense all action 
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=ce_over --model_training=super_dense_all_action --window_eval=true --uniform_ss=false --window_size=7
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=sampled_softmax --model_training=super_dense_all_action --window_eval=true --uniform_ss=true --window_size=7

#future_rolling 
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=ce_over --model_training=future_rolling --window_eval=true --uniform_ss=false --window_size=7
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=sampled_softmax --model_training=future_rolling --window_eval=true --uniform_ss=true --window_size=7

# combined
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda  --model_training=combined --loss_type=ce_over --window_eval=true --window_size=7
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda  --model_training=combined --loss_type=sampled_softmax --uniform_ss=true --window_eval=true --window_size=7
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda  --model_training=combined --loss_type=sampled_softmax --uniform_ss=false --window_eval=true --window_size=7

# Second Exp:
# T2V
#python main.py --dataset=ml-1m_time --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --loss_type=ce_over --model_training=combined --window_eval=true --uniform_ss=false --window_size=7  --temporal=true
#python main.py --dataset=ml-1m_time --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --loss_type=sampled_softmax --model_training=combined --window_eval=true --uniform_ss=true --window_size=7 --temporal=true

# AR
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --loss_type=ce_over --model_training=combined --window_eval=true --uniform_ss=false --strategy=autoregressive --window_size=7
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --loss_type=sampled_softmax --model_training=combined --window_eval=true --uniform_ss=true --strategy=autoregressive --window_size=7

# TF
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --loss_type=ce_over --model_training=combined --window_eval=true --uniform_ss=false --strategy=teacher_forcing --window_size=7
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --loss_type=sampled_softmax --model_training=combined --window_eval=true --uniform_ss=true --strategy=teacher_forcing --window_size=7

# masking
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=ce_over --model_training=combined --window_eval=true --uniform_ss=false --masking=true
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=sampled_softmax --model_training=combined --window_eval=true --uniform_ss=true --masking=true

# masking + TF
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=ce_over --model_training=combined --strategy=teacher_forcing --window_eval=true --uniform_ss=false --masking=true


# Third Exp:
# combined window 3
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda  --model_training=combined --loss_type=sampled_softmax --uniform_ss=true --window_eval=true --window_size=3 --window_eval_size=3

# base window 10
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda  --model_training=combined --loss_type=sampled_softmax --uniform_ss=true --window_eval=true --window_size=10 --window_eval_size=10

# base window 14
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda  --model_training=combined --loss_type=sampled_softmax --uniform_ss=true --window_eval=true --window_size=14 --window_eval_size=14

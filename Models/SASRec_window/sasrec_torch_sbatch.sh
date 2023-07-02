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


source /home/${STUDENT_ID}/miniconda3/bin/activate mscproject

#python main.py --device=cuda --dataset=Video --train_dir=default
# FOR TRAINING with BCE loss
#python main.py --dataset=ml-1m --train_dir=default --maxlen=199 --dropout_rate=0.2 --device=cuda --data_partition=None --model_training=None --window_eval=true # BASELINE
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=independent --model_training=None --window_eval=true   # sasrec independent            
# python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --model_training=all_action --window_eval=true   # all action   
#python main.py --dataset=ml-1m --train_dir=default --maxlen=201 --dropout_rate=0.2 --device=cuda --data_partition=None --model_training=dense_all_action --window_eval=true  # dense all action 
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --model_training=super_dense_all_action --window_eval=true  # super dense all action 
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=teacher_forcing --model_training=None --window_eval=true   # sasrec teacher forcing            


# FOR TRAINING with Sampled softmax loss
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=sampled_softmax --model_training=None --window_eval=true # BASELINE
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=independent --loss_type=sampled_softmax --model_training=None --window_eval=true   # sasrec independent            
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=sampled_softmax --model_training=all_action --window_eval=true   # all action   
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=sampled_softmax --model_training=dense_all_action --window_eval=true  # dense all action 
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=sampled_softmax --model_training=super_dense_all_action --window_eval=true  # super dense all action 
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=teacher_forcing --loss_type=sampled_softmax --model_training=None --window_eval=true

# FOR INFERENCE ONLY (NO TRAINING)
# python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --data_partition=None --model_training=None --window_eval=true --inference_only=true # BASELINE
# python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=independent --model_training=None --window_eval=true --inference_only=true  # sasrec independent            
# python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --model_training=all_action --window_eval=true --inference_only=true # all action   
# python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --model_training=dense_all_action --window_eval=true --inference_only=true # dense all action        


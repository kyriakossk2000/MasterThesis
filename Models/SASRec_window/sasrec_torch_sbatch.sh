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
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --model_training=all_action --window_eval=true   # all action   
#python main.py --dataset=ml-1m --train_dir=default --maxlen=201 --dropout_rate=0.2 --device=cuda --data_partition=None --model_training=dense_all_action --window_eval=true  # dense all action 
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --model_training=super_dense_all_action --window_eval=true  # super dense all action 
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=teacher_forcing --model_training=None --window_eval=true   # sasrec teacher forcing            
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --model_training=future_rolling --loss_type=bce --window_eval=true --window_size=7 # future rolling with bce

# Masking
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --model_training=None --loss_type=bce --window_eval=true --window_size=7 --masking=true --mask_prob=0.15
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=independent --model_training=None --loss_type=bce --window_eval=true --window_size=7 --masking=true --mask_prob=0.15
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=teacher_forcing --model_training=None --loss_type=bce --window_eval=true --window_size=7 --masking=true --mask_prob=0.15


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FOR TRAINING with Sampled softmax loss
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=sampled_softmax --model_training=None --window_eval=true # BASELINE
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=independent --loss_type=sampled_softmax --model_training=None --window_eval=true   # sasrec independent            
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=sampled_softmax --model_training=all_action --window_eval=true   # all action   
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=sampled_softmax --model_training=dense_all_action --window_eval=true  # dense all action 
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=sampled_softmax --model_training=super_dense_all_action --window_eval=true  # super dense all action 
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=teacher_forcing --loss_type=sampled_softmax --model_training=None --window_eval=true
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --model_training=future_rolling --loss_type=sampled_softmax --window_eval=true --window_size=7 # future rolling with sampled softmax
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --model_training=future_rolling --loss_type=sampled_softmax --window_eval=true --window_size=7 --uniform_ss=true
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --model_training=future_rolling --loss_type=sampled_softmax --window_eval=true --window_size=7 --uniform_ss=true --strategy=teacher_forcing

# Masking
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --model_training=None --loss_type=sampled_softmax --window_eval=true --window_size=7 --masking=true --mask_prob=0.15
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=independent --model_training=None --loss_type=sampled_softmax --window_eval=true --window_size=7 --masking=true --mask_prob=0.15
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=teacher_forcing --model_training=None --loss_type=sampled_softmax --window_eval=true --window_size=7 --masking=true --mask_prob=0.15
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=sampled_softmax --model_training=all_action --window_eval=true --strategy=autoregressive
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=sampled_softmax --model_training=all_action --window_eval=true --strategy=teacher_forcing --uniform_ss=true


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FOR TRAINING with CE_over loss
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --model_training=future_rolling --loss_type=ce_over --window_eval=true --window_size=7  # future rolling with ce-over
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=ce_over --model_training=future_rolling --window_eval=true --strategy=autoregressive
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=ce_over --model_training=future_rolling --window_eval=true --strategy=teacher_forcing

#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --model_training=all_action --loss_type=ce_over --window_eval=true --window_size=7
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=ce_over --model_training=all_action --window_eval=true --strategy=autoregressive
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=ce_over --model_training=all_action --window_eval=true --strategy=teacher_forcing

# masking
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --model_training=all_action --loss_type=ce_over --window_eval=true --window_size=7 --masking=true --mask_prob=0.25

# TIME2VEC EXP
python main.py --dataset=ml-1m_time --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=ce_over --model_training=all_action --window_eval=true --temporal=true  # AA + T2V CE
#python main.py --dataset=ml-1m_time --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=ce_over --model_training=all_action --window_eval=true --masking=true --temporal=true  # AA + T2V + MASK CE
#python main.py --dataset=ml-1m_time --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=ce_over --model_training=all_action --window_eval=true --strategy=autoregressive --temporal=true  # AA + T2V + AR CE
#python main.py --dataset=ml-1m_time --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=ce_over --model_training=all_action --window_eval=true --strategy=teacher_forcing --temporal=true  # AA + T2V + TF CE

#python main.py --dataset=ml-1m_time --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=sampled_softmax --model_training=all_action --window_eval=true  --uniform_ss=true --temporal=true   # AA + T2V SS Uni
#python main.py --dataset=ml-1m_time --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=sampled_softmax --model_training=all_action --window_eval=true --strategy=autoregressive --uniform_ss=true --temporal=true   # AA + T2V + AR SS Uni
#python main.py --dataset=ml-1m_time --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=sampled_softmax --model_training=all_action --window_eval=true --strategy=teacher_forcing --uniform_ss=true --temporal=true   # AA + T2V + TF SS Uni

#python main.py --dataset=ml-1m_time --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --loss_type=sampled_softmax --model_training=all_action --window_eval=true  --uniform_ss=false --temporal=true  # AA + T2V SS LogQ



# FOR INFERENCE ONLY (NO TRAINING)
# python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --data_partition=None --model_training=None --window_eval=true --inference_only=true # BASELINE
# python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=independent --model_training=None --window_eval=true --inference_only=true  # sasrec independent            
# python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --model_training=all_action --window_eval=true --inference_only=true # all action   
# python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --data_partition=None --model_training=dense_all_action --window_eval=true --inference_only=true # dense all action        


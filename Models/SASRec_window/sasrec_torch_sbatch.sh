#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:titan-x:1
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

#python main.py --device=cuda --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2
#python main.py --device=cuda --dataset=Video --train_dir=default
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --window_split=false --window_eval=true --device=cuda
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --window_split=true --window_eval=true --device=cuda --num_epochs=100
#python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --window_split=false --window_eval=true --inference_only=true --state_dict_path=SASRec_dict_baselien_no_split.pth  --device=cuda
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --window_split=true --window_eval=true --all_action=true --device=cuda

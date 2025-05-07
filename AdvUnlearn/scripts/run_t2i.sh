#!/bin/bash
#SBATCH -J run_t2i # 작업이름
#SBATCH -p cas_v100nv_8 # 파티션
#SBATCH --nodes=1 # 
#SBATCH --ntasks-per-node=1
#SBATCH -o ../out/%x_%j.out # 아웃풋 경로
#SBATCH -e ../err/%x_%j.err # 에러경로
#SBATCH --time=48:00:00 
#SBATCH --gres=gpu:1 # using 2 gpus per node
#SBATCH --comment pytorch
module purge
python run_t2i_out.py
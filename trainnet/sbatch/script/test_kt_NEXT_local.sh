#!/usr/bin/env bash
#SBATCH -A SNIC2022-5-184 -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A100:1
#SBATCH -t 2-48:00:00
echo ""
echo "This job was started--$@"
echo ""

SAVE_PATH=/home/liu/project/TrainNet/data
CODE=/home/liu/project/TrainNet
SIF=/home/liu/project/TrainNet/trainnet.sif


singularity exec --nv \
--pwd /workspace/TrainNet/tools \
--bind ${CODE}/data:/workspace/TrainNet/data \
--bind ${CODE}/output:/workspace/TrainNet/output \
--bind ${CODE}/tools:/workspace/TrainNet/tools \
--bind ${CODE}/pcdet:/workspace/TrainNet/pcdet \
--bind ${CODE}/cfgs:/workspace/TrainNet/cfgs \
${SIF} \
python3 ${CODE}/tools/test.py \
--cfg_file ${CODE}/cfgs/kt_NEXT.yaml \
--ckpt /home/liu/project/TrainNet/output/20220829-224838/ckpt/checkpoint_epoch_1.pth \
--batch_size 1 \
--save_path SAVE_PATH \
"$@"




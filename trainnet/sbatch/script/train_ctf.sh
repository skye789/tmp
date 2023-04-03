#!/usr/bin/env bash
#SBATCH -A SNIC2022-5-184 -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A100:1
#SBATCH -t 2-48:00:00
echo ""
echo "This job was started--$@"
echo ""

SAVE_PATH=/cephyr/NOBACKUP/groups/snic2021-7-127/yucong/TrainNet/data
CODE=/cephyr/NOBACKUP/groups/snic2021-7-127/yucong/TrainNet
SIF=/cephyr/NOBACKUP/groups/snic2021-7-127/yucong/TrainNet/trainnet.sif


singularity exec --nv \
--pwd /workspace/TrainNet/tools \
--bind ${CODE}/data:/workspace/TrainNet/data \
--bind ${CODE}/output:/workspace/TrainNet/output \
--bind ${CODE}/tools:/workspace/TrainNet/tools \
--bind ${CODE}/pcdet:/workspace/TrainNet/pcdet \
--bind ${CODE}/cfgs:/workspace/TrainNet/cfgs \
${SIF} \
python3 ${CODE}/tools/train.py \
--cfg_file ${CODE}/cfgs/ctf.yaml \
--batch_size 1 \
--save_path SAVE_PATH \
--workers 8
"$@"




#!/bin/sh
CODE=/home/liu/project/TrainNet
nvidia-docker run -it\
 -v ${CODE}:/workspace/TrainNet \
trainnet:latest /bin/bash


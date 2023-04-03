#!/bin/sh

# Build docker image
#docker build -t trainnet .

# Convert to singularity
singularity build trainnet.sif docker-daemon://trainnet:latest

# nvidia-docker run -it trainnet:latest /bin/bash


# nvidia-docker run -it -v /home/liu/project/TrainNet:/workspace/TrainNet \
# trainnet:latest /bin/bash


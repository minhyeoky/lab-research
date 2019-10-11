#!/usr/bin/env bash
gpu=$2
export docker_dir=/tf
docker build . -t research
docker run -it --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=${gpu} -e docker_dir=${docker_dir} -v /home/mhlee:${docker_dir} \
research \
python /tf/research/train.py --config=${docker_dir}/data/experiment/config/$1 --data_dir=${docker_dir}/data --output_dir=${docker_dir}/data/experiment/output
#!/usr/bin/env bash
export docker_dir=/tf
docker build . -t research
docker run -it --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=1 -e docker_dir=${docker_dir} -v /home/mhlee:${docker_dir} \
research \
python /tf/research/train.py --config=${docker_dir}/research/config.json --data_dir=${docker_dir}/data --output_dir=${docker_dir}/data/experiment/output
#CUDA_VISIBLE_DEVICES=0 python train.py --config=./config.json --data=../data --output_path=../data/experiment/output
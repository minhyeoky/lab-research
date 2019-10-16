#!/usr/bin/env bash
gpu=$2
infer_audio=$3
export docker_dir=/tf
docker build . -t research
docker run -it --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=${gpu} -e docker_dir=${docker_dir} -v /home/mhlee:${docker_dir} \
research \
python /tf/research/train.py --infer=True --infer_audio=${docker_dir}/${infer_audio} --config=${docker_dir}/research/config/$1 --data_dir=${docker_dir}/data --output_dir=${docker_dir}/data/experiment/output

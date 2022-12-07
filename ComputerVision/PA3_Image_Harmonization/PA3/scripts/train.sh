#!/usr/bin/env bash
#--dataset_root /jisu/dataset/iHarmony4resized256/ \

python train.py \
--dataset_root /usr/working_env/preprocessed \
--name experiment_train_plus \
--checkpoints_dir ./checkpoints/scratch/ \
--model rainnet \
--netG rainnet \
--dataset_mode iharmony4 \
--is_train 1 \
--gan_mode wgangp \
--normD instance \
--normG RAIN \
--preprocess None \
--niter 100 \
--niter_decay 100 \
--input_nc 3 \
--batch_size 12 \
--num_threads 6 \
--lambda_L1 100 \
--print_freq 400 \
--gpu_ids 1 \
#--continue_train \
#--load_iter 87 \
#--epoch 88 \

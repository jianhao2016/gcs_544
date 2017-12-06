#! /bin/sh
#
# group_gcs.sh
# Copyright (C) 2017 qizai <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.
#



JOB_NAME='project544_s_05'
BUCKET_NAME='gs://544projects-cloud'
gcloud ml-engine jobs submit training ${JOB_NAME} \
    --package-path models \
    --module-name models.cifar10_train_s05 \
    --staging-bucket ${BUCKET_NAME} \
    --job-dir ${BUCKET_NAME}/${JOB_NAME} \
    --runtime-version 1.2 \
    --region us-east1 \
    --config cloudml-4gpu.yaml \
    -- \
    --train_data_dir ${BUCKET_NAME}/data/cifar10_train.tfrecords \
    --test_data_dir ${BUCKET_NAME}/data/cifar10_test.tfrecords \
    --summaries_dir ${BUCKET_NAME}/results/summaries_${JOB_NAME}/ \
    --model_dir ${BUCKET_NAME}/results/saved_models_${JOB_NAME}/ \
    --nEpochs 100 \
    --batch_size 128 \
    --LR 1e-1 \
    --weightDecay 1e-4 \
    --depth 20 \
    --full 512 \
    --numChannels 128 \
    --sparsity 0.5 \
    --number_of_b 512 \
    --convSize 3 \
    --momentum 0.9 \
    --shared_weights False \

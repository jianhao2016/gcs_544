#! /bin/sh
#
# gcs.sh
# Copyright (C) 2017 qizai <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.
#


JOB_NAME='project544_Dec_05_UseTFRecord_to_load'
gcloud ml-engine jobs submit training ${JOB_NAME} \
    --package-path models \
    --module-name models.cifar10_train \
    --staging-bucket gs://544projects-cloud \
    --job-dir gs://544projects-cloud/${JOB_NAME} \
    --runtime-version 1.2 \
    --region us-east1 \
    --config cloudml-4gpu.yaml \
    -- \
    --train_data_dir gs://544projects-cloud/data/cifar10_train.tfrecords \
    --test_data_dir gs://544projects-cloud/data/cifar10_test.tfrecords \
    --depth 20

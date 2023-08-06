#!/bin/bash

#cd ../..

# custom config
DATA=DATA/
TRAINER=cshp
# TRAINER=CoOp

DATASET=oxford_pets
SEED=42

CFG=vit_b16_c4_ep10_batch1
# CFG=vit_b16_ctxv1  # uncomment this when TRAINER=CoOp
SHOTS=16
LOADEP=10
SUB=all

COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
DIR=output/base2new/test_${SUB}/${COMMON_DIR}

python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
--model-dir ${MODEL_DIR} \
--load-epoch ${LOADEP} \
--eval-only \
DATASET.NUM_SHOTS ${SHOTS} \
DATASET.SUBSAMPLE_CLASSES ${SUB}

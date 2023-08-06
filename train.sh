#!/bin/bash

#cd ../..

# custom config
DATA=DATA/
# TRAINER=CoOp
DATASET=oxford_flowers
SEED=42
CFG=vit_b16_c4_ep10_batch1
# CFG=vit_b16_ctxv1  # uncomment this when TRAINER=CoOp
# CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet
SHOTS=16
DATA_PROMPT=oxford_flowers
for TRAINER in 'cshp' 'CoCoOp'
do
  #训练
  DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
  python train.py \
  --root ${DATA} \
  --seed ${SEED} \
  --trainer ${TRAINER} \
  --dataset-config-file configs/datasets/${DATASET}.yaml \
  --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
  --output-dir ${DIR} \
  --dataset-name ${DATA_PROMPT} \
  DATASET.NUM_SHOTS ${SHOTS} \
  DATASET.SUBSAMPLE_CLASSES base
  #测试new
  LOADEP=10
  SUB=new
  COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
  MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
  DIR=output/base2new/test_${SUB}/${COMMON_DIR}
#  python train.py \
#  --root ${DATA} \
#  --seed ${SEED} \
#  --trainer ${TRAINER} \
#  --dataset-config-file configs/datasets/${DATASET}.yaml \
#  --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
#  --output-dir ${DIR} \
#  --model-dir ${MODEL_DIR} \
#  --load-epoch ${LOADEP} \
#  --eval-only \
#  --dataset-name ${DATA_PROMPT} \
#  DATASET.NUM_SHOTS ${SHOTS} \
#  DATASET.SUBSAMPLE_CLASSES ${SUB}

  #测试all
#  SUB=all
#  python train.py \
#  --root ${DATA} \
#  --seed ${SEED} \
#  --trainer ${TRAINER} \
#  --dataset-config-file configs/datasets/${DATASET}.yaml \
#  --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
#  --output-dir ${DIR} \
#  --model-dir ${MODEL_DIR} \
#  --load-epoch ${LOADEP} \
#  --eval-only \
#  DATASET.NUM_SHOTS ${SHOTS} \
#  DATASET.SUBSAMPLE_CLASSES ${SUB}
done

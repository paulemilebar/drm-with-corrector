python3 main.py --base configs/correction/train.yaml \
    --train --devices 0, --name CORRECTOR_SMOKE --scale_lr false \
    --limit_train_batches 2 --limit_val_batches 1 \
    --max_epochs 1 --num_sanity_val_steps 0
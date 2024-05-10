## Render Watercolor


Run from inside build directory.

Regular pbrt:

```
./pbrt ../../scenes/watercolor/watercolor/camera-1.pbrt --outfile watercolor_camera_1.exr --spp 100
```

Multiview pbrt test:

```
./pbrt_multiview ../../scenes/watercolor/watercolor/camera-1.pbrt --outfile watercolor.exr --spp 100
```

## Train Models

Train baseline RGB denoiser:

```
python train.py \
    device=mps \
    logging.eval_interval=100 \
    logging.log_interval=100 \
    logging.ckpt_interval=1350 \
    data.num_dataloader_workers=1 \
    logging.save_ckpt=True
```

Train full features UNET denoiser:

```
python train.py \
    device=mps \
    logging.eval_interval=100 \
    logging.log_interval=100 \
    logging.ckpt_interval=1350 \
    data.num_dataloader_workers=1 \
    logging.save_ckpt=False \
    wandb.mode=disabled \
    model=full_features_unet
```

## Measure Throughput

Measure full features UNET denoiser throughput:

```
python eval.py --config-name pbrt \
    device=mps \
    model=full_features_unet \
    logging.ckpt_dir=../checkpoints/full_features_unet/05_09_2024-15_39_42/iter_9.pt \
    data=dummy_pbrt \
    data.num_dataloader_workers=1 \
    logging.save_ckpt=False \
    wandb.mode=disabled \
    model=full_features_unet \
    data.batch_size=4000
```

## Visualize outputs

```
python eval.py \
    device=mps \
    data.num_dataloader_workers=1 \
    logging.ckpt_dir=../checkpoints/rgb_baseline/05_04_2024-17_06_31/iter_1349.pt
```
## Render Watercolor


Run from inside build directory.

Regular pbrt:

```
./pbrt ../../scenes/watercolor/watercolor/camera-1.pbrt --outfile watercolor_camera_1.exr --spp 100
```

Regular pbrt teapot:

```
./pbrt ../../scenes/teapot/scene-v4.pbrt --outfile test.png --spp 100
```

Multiview pbrt test:

```
./pbrt_multiview ../../scenes/watercolor/watercolor/camera-1.pbrt --outfile watercolor.exr --spp 100
```

## Train Models

### PBRT

Full features UNET denoiser sweeps:

```
python train.py --multirun --config-name pbrt \
    device=cuda \
    logging.eval_interval=100 \
    logging.log_interval=10 \
    logging.ckpt_interval=100 \
    data.num_dataloader_workers=1 \
    model=full_features_unet \
    model.loss_name=l1_error,l2_error \
    optimizer.lr=1e-1,1e-2,1e-3 \
    logging.save_ckpt=False \
    wandb.mode=disabled
```

Visualize predictions:

```
python eval.py --config-name pbrt \
    device=cuda \
    logging.eval_interval=100 \
    logging.log_interval=1 \
    logging.ckpt_interval=100 \
    logging.ckpt_dir=../checkpoints/05_19_2024-06_04_07 \
    data.num_dataloader_workers=1 \
    logging.save_ckpt=False \
    wandb.mode=disabled \
    model=full_features_unet
```

### Tiny Imagenet

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
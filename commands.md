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

## Bistro

```
python train.py --multirun --config-name train_blender \
    device=mps \
    logging.eval_interval=100 \
    logging.log_interval=100 \
    logging.ckpt_interval=100 \
    logging.ckpt_dir=../checkpoints/amazon/full_features_unet \
    data=bistro \
    data.low_spp=1 \
    data.high_spp=512 \
    data.num_dataloader_workers=1 \
    model=full_features_unet \
    model.loss_name=l1_error \
    optimizer.lr=1e-2 \
    training.num_grad_steps=5000 \
    data.batch_size=128 \
    logging.save_ckpt=True \
    wandb.mode=online \
    wandb.run_name_suffix=rgb-albedo-depth-surface_normals
```

## Classroom SPP Sweeps

```
python train.py --multirun --config-name train_blender \
    device=mps \
    logging.eval_interval=100 \
    logging.log_interval=100 \
    logging.ckpt_interval=100 \
    logging.ckpt_dir=../checkpoints/classroom/full_features_unet/spp_sweeps \
    data=classroom \
    data.low_spp=1,4,8 \
    data.high_spp=512 \
    data.num_dataloader_workers=1 \
    model=full_features_unet \
    model.loss_name=l1_error \
    optimizer.lr=1e-2 \
    training.num_grad_steps=5000 \
    data.batch_size=128 \
    logging.save_ckpt=True \
    wandb.mode=online \
    wandb.run_name_suffix=rgb-albedo-depth-surface_normals
```

## Classroom SPP Sweeps

```
python train.py --multirun --config-name train_blender \
    device=mps \
    logging.eval_interval=100 \
    logging.log_interval=100 \
    logging.ckpt_interval=100 \
    logging.ckpt_dir=../checkpoints/classroom/full_features_unet/spp_sweeps \
    data=classroom \
    data.low_spp=1,4,8 \
    data.high_spp=512 \
    data.num_dataloader_workers=1 \
    model=full_features_unet \
    model.loss_name=l1_error \
    optimizer.lr=1e-2 \
    training.num_grad_steps=5000 \
    data.batch_size=128 \
    logging.save_ckpt=True \
    wandb.mode=online \
    wandb.run_name_suffix=rgb-albedo-depth-surface_normals
```


## Evals

Benchmark throughput:

```
python eval.py --config-name eval_blender \
    device=cuda \
    logging.ckpt_dir=../checkpoints/classroom/full_features_unet/rgb-diffuse-depth-surface_normals/05_28_2024-23_27_55/iter_4999.pt \
    data=classroom \
    data.low_spp=1 \
    data.num_dataloader_workers=1 \
    num_samples=1024 \
    num_warmup=40 \
    num_trials=40 \
    data.batch_size=1024 \
    logging.save_ckpt=False \
    wandb.mode=disabled \
    model=full_features_unet
```

Visualize predictions:

```
python eval.py --config-name eval_blender \
    device=mps \
    logging.ckpt_dir=../checkpoints/classroom/full_features_unet/rgb-diffuse-depth-surface_normals/05_28_2024-23_27_55/iter_4999.pt \
    data=classroom \
    data.num_dataloader_workers=1 \
    data.batch_size=64 \
    model=full_features_unet
```

Compute errors on val and test splits:

```
python eval.py --config-name eval_blender \
    device=mps \
    logging.ckpt_dir=../checkpoints/classroom/full_features_unet/rgb-diffuse-depth-surface_normals/05_28_2024-23_27_55/iter_4999.pt \
    data=classroom \
    data.low_spp=4 \
    data.num_dataloader_workers=1 \
    data.batch_size=64 \
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
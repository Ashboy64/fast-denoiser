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

### San Miguel

```
python train.py --multirun --config-name train_pbrt \
    device=mps \
    logging.eval_interval=100 \
    logging.log_interval=100 \
    logging.ckpt_interval=100 \
    logging.ckpt_dir=../checkpoints/san_miguel/full_features_unet \
    data=san_miguel \
    data.low_spp=1 \
    data.high_spp=1024 \
    data.num_dataloader_workers=1 \
    model=full_features_unet \
    model.loss_name=l1_error \
    optimizer.lr=1e-2 \
    training.num_grad_steps=5000 \
    data.batch_size=128 \
    logging.save_ckpt=False \
    wandb.mode=disabled \
    wandb.run_name_suffix=rgb-albedo-depth-surface_normals
```

### Barbershop

```
python train.py --multirun --config-name train_blender \
    device=mps \
    logging.eval_interval=100 \
    logging.log_interval=100 \
    logging.ckpt_interval=100 \
    logging.ckpt_dir=../checkpoints/barbershop/full_features_unet \
    data=barbershop \
    data.low_spp=1 \
    data.high_spp=1024 \
    data.num_dataloader_workers=1 \
    model=full_features_unet \
    model.loss_name=l1_error \
    optimizer.lr=1e-2 \
    training.num_grad_steps=5000 \
    data.batch_size=128 \
    logging.save_ckpt=False \
    wandb.mode=disabled \
    wandb.run_name_suffix=rgb-albedo-depth-surface_normals
```

### Classroom SPP Sweeps

```
python train.py --multirun --config-name train_blender \
    device=mps \
    logging.eval_interval=100 \
    logging.log_interval=100 \
    logging.ckpt_interval=100 \
    logging.ckpt_dir=../checkpoints/classroom/full_features_unet/spp_sweeps \
    data=classroom \
    data.low_spp=1,4,8 \
    data.high_spp=1024 \
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

### Classroom Feature Ablations

```
python train.py --multirun --config-name train_blender \
    device=mps \
    logging.eval_interval=100 \
    logging.log_interval=100 \
    logging.ckpt_interval=100 \
    logging.ckpt_dir=../checkpoints/classroom/full_features_unet/feature_ablations/rgb_albedo \
    data=classroom \
    data.low_spp=1 \
    data.high_spp=1024 \
    data.num_dataloader_workers=1 \
    model=full_features_unet \
    model.loss_name=l1_error \
    optimizer.lr=1e-2 \
    training.num_grad_steps=5000 \
    data.batch_size=128 \
    logging.save_ckpt=True \
    wandb.mode=online \
    wandb.run_name_suffix=rgb-albedo
```

### Bistro

```
python train.py --multirun --config-name train_blender \
    device=mps \
    logging.eval_interval=100 \
    logging.log_interval=100 \
    logging.ckpt_interval=100 \
    logging.ckpt_dir=../checkpoints/bistro/full_features_unet \
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

### Classroom Bistro Hybrid

```
python train.py --multirun --config-name train_blender \
    device=mps \
    logging.eval_interval=100 \
    logging.log_interval=100 \
    logging.ckpt_interval=100 \
    logging.ckpt_dir=../checkpoints/hybrid/full_features_unet \
    data=hybrid_blender \
    data.num_dataloader_workers=1 \
    data.batch_size=128 \
    model=full_features_unet \
    model.loss_name=l1_error \
    optimizer.lr=1e-2 \
    training.num_grad_steps=5000 \
    logging.save_ckpt=False \
    wandb.mode=disabled \
    wandb.run_name_suffix=rgb-albedo-depth-surface_normals
```

## Evals

### Scene Transfer

```
python eval.py --config-name eval_blender \
    device=mps \
    logging.ckpt_dir=../checkpoints/classroom/full_features_unet/spp_sweeps/06_01_2024-14_35_17/iter_4999.pt \
    predictions_dir=../denoiser-outputs/scene_transfer/classroom_to_classroom \
    data=classroom \
    data.low_spp=1 \
    data.high_spp=1024 \
    data.num_dataloader_workers=1 \
    compute_errors=True \
    visualize_predictions=True \
    measure_throughput=False \
    num_samples=1024 \
    num_warmup=40 \
    num_trials=40 \
    data.batch_size=1024 \
    logging.save_ckpt=False \
    wandb.mode=disabled \
    model=full_features_unet
```

### Classroom

```
python eval.py --config-name eval_blender \
    device=mps \
    logging.ckpt_dir=../checkpoints/classroom/full_features_unet/feature_ablations/rgb_albedo_depth/06_01_2024-16_58_47/iter_4999.pt \
    predictions_dir=../denoiser-outputs/classroom/feature_ablations/rgb_albedo_depth \
    data=classroom \
    data.low_spp=1 \
    data.num_dataloader_workers=1 \
    compute_errors=True \
    visualize_predictions=True \
    measure_throughput=False \
    num_samples=1024 \
    num_warmup=40 \
    num_trials=40 \
    data.batch_size=1024 \
    logging.save_ckpt=False \
    wandb.mode=disabled \
    model=full_features_unet
```

### Barbershop

```
python eval.py --config-name eval_blender \
    device=mps \
    logging.ckpt_dir=../checkpoints/barbershop/full_features_unet/06_01_2024-15_47_08/iter_4999.pt \
    predictions_dir=../denoiser-outputs/barbershop/spp_1 \
    data=barbershop \
    data.low_spp=1 \
    data.high_spp=1024 \
    data.num_dataloader_workers=1 \
    compute_errors=True \
    visualize_predictions=True \
    measure_throughput=False \
    num_samples=1024 \
    num_warmup=40 \
    num_trials=40 \
    data.batch_size=1024 \
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
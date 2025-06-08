# SiT-XL E2E SDXL

End-to-end training implementation of SiT-XL (Scalable interpolant Transformer) for SDXL-compatible image generation. This implementation trains SiT-XL, CLIP encoders, and VAE simultaneously, similar to REPA-E.

## Features

- **SiT-XL Architecture**: Transformer-based diffusion model with 2-3B parameters
- **SDXL Compatibility**: Works with SDXL VAE and dual CLIP text encoders
- **End-to-End Training**: Simultaneous training of SiT-XL, VAE, and CLIP encoders
- **Aspect Ratio Bucketing**: Efficient training with various aspect ratios
- **Multiple Schedulers**: Support for DDIM, DDPM, Euler, Euler-A, DPM, etc.

## Model Architecture

- **SiT-XL**: Scalable interpolant Transformer replacing U-Net
  - 2B variant: 1536 dim, 42 layers, 24 heads
  - 3B variant: 1792 dim, 48 layers, 28 heads
- **VAE**: SDXL VAE (madebyollin/sdxl-vae-fp16-fix)
- **Text Encoders**: 
  - CLIP-L (OpenAI CLIP ViT-L/14)
  - CLIP-G (OpenCLIP ViT-bigG/14)

## Requirements

```bash
pip install torch torchvision
pip install transformers diffusers accelerate
pip install einops tqdm tensorboard
pip install pillow numpy
```

Note: Pillow (PIL) includes WebP support by default in most installations.

## Dataset Format

The dataset should follow the same format as [tagutl](https://github.com/celll1/tagutl):
- Images: `.png`, `.jpg`, `.jpeg`, or `.webp` files
- Tags: `.txt` files with the same name as images
- Optional metadata: `.json` files

Directory structure:
```
data/
├── image1.png
├── image1.txt
├── image1.json (optional)
├── image2.jpg
├── image2.txt
├── photo.jpeg
├── photo.txt
├── artwork.webp
├── artwork.txt
└── ...
```

## Training

Basic training command:
```bash
python train.py \
    --data_root /path/to/dataset \
    --output_dir ./output \
    --model_size 2b \
    --batch_size 1 \
    --learning_rate 1e-4 \
    --num_epochs 100 \
    --use_aspect_ratio_bucket \
    --mixed_precision \
    --use_ema
```

For end-to-end training (including VAE and text encoders):
```bash
python train.py \
    --data_root /path/to/dataset \
    --output_dir ./output \
    --model_size 2b \
    --train_vae \
    --train_text_encoder \
    --batch_size 1 \
    --learning_rate 1e-4 \
    --num_epochs 100 \
    --use_aspect_ratio_bucket \
    --mixed_precision \
    --use_ema
```

### Training Arguments

- `--model_size`: Model variant (`2b` or `3b`)
- `--data_root`: Path to training dataset
- `--output_dir`: Directory for checkpoints and logs
- `--batch_size`: Training batch size
- `--learning_rate`: Initial learning rate
- `--num_epochs`: Number of training epochs
- `--use_aspect_ratio_bucket`: Enable aspect ratio bucketing
- `--train_vae`: Train VAE along with SiT-XL
- `--train_text_encoder`: Train CLIP encoders
- `--mixed_precision`: Use mixed precision training
- `--use_ema`: Use exponential moving average

## Inference

Generate images using trained model:
```bash
python inference.py \
    --checkpoint ./output/final/sit_model.pt \
    --model_size 2b \
    --prompt "a beautiful landscape painting" \
    --negative_prompt "low quality, blurry" \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --scheduler euler \
    --output_dir ./outputs
```

### Inference Arguments

- `--checkpoint`: Path to model checkpoint
- `--prompt`: Text prompt for generation
- `--negative_prompt`: Negative prompt for guidance
- `--height`, `--width`: Image dimensions (default 1024x1024)
- `--num_inference_steps`: Number of denoising steps
- `--guidance_scale`: Classifier-free guidance scale
- `--scheduler`: Noise scheduler type (ddim, euler, euler_a, etc.)
- `--seed`: Random seed for reproducibility

## Model Checkpoints

Checkpoints are saved during training:
- `checkpoint-{step}/`: Intermediate checkpoints
  - `sit_model.pt`: SiT-XL weights
  - `vae.pt`: VAE weights (if training VAE)
  - `text_encoder_l.pt`, `text_encoder_g.pt`: CLIP weights (if training)
  - `ema_model.pt`: EMA weights (if using EMA)
  - `optimizer.pt`, `scheduler.pt`: Training state
- `final/`: Final model checkpoint

## References

- [SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers](https://github.com/willisma/SiT)
- [SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://stability.ai/stable-diffusion)
- [REPA: Bridging the Gap between Pixels and Perception](https://github.com/End2End-Diffusion/REPA-E)

## License

This project is for research purposes. Please check the licenses of the individual components:
- SiT: MIT License
- SDXL components: CreativeML Open RAIL++-M License
- CLIP models: MIT License

## Notes

- The model predicts noise (epsilon prediction) similar to SD/SDXL, not using Rectified Flow
- All components (SiT-XL, VAE, CLIP) can be trained simultaneously for end-to-end optimization
- Aspect ratio bucketing is recommended for efficient training with SDXL's various resolutions
- Mixed precision training is recommended for memory efficiency

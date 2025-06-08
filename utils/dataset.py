"""
Dataset utilities for loading images and tag files with aspect ratio bucketing.
Based on https://github.com/celll1/tagutl dataset format.
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
from transformers import CLIPTokenizer


class AspectRatioBucket:
    """Manages aspect ratio buckets for efficient batching."""
    
    def __init__(
        self,
        base_resolution: int = 1024,
        bucket_resolutions: Optional[List[Tuple[int, int]]] = None,
        tolerance: float = 0.1,
    ):
        self.base_resolution = base_resolution
        self.tolerance = tolerance
        
        # Default SDXL bucket resolutions
        if bucket_resolutions is None:
            self.bucket_resolutions = [
                (512, 2048), (576, 1792), (640, 1536), (704, 1344),
                (768, 1280), (832, 1216), (896, 1152), (960, 1088),
                (1024, 1024),  # Square
                (1088, 960), (1152, 896), (1216, 832), (1280, 768),
                (1344, 704), (1536, 640), (1792, 576), (2048, 512),
            ]
        else:
            self.bucket_resolutions = bucket_resolutions
        
        # Precompute bucket aspect ratios
        self.bucket_aspects = [(w/h) for w, h in self.bucket_resolutions]
        self.buckets = defaultdict(list)
    
    def get_bucket_resolution(self, width: int, height: int) -> Tuple[int, int]:
        """Find the best matching bucket resolution for given dimensions."""
        aspect_ratio = width / height
        
        # Find closest aspect ratio
        min_diff = float('inf')
        best_bucket = self.bucket_resolutions[len(self.bucket_resolutions)//2]  # Default to square
        
        for i, bucket_aspect in enumerate(self.bucket_aspects):
            diff = abs(aspect_ratio - bucket_aspect)
            if diff < min_diff:
                min_diff = diff
                best_bucket = self.bucket_resolutions[i]
        
        return best_bucket
    
    def add_item(self, item_id: str, width: int, height: int):
        """Add an item to the appropriate bucket."""
        bucket = self.get_bucket_resolution(width, height)
        self.buckets[bucket].append(item_id)
    
    def get_batch_indices(self, batch_size: int) -> Dict[Tuple[int, int], List[str]]:
        """Get indices for a batch, grouped by bucket."""
        batch_buckets = defaultdict(list)
        
        # Collect items from buckets
        available_buckets = [b for b in self.buckets if len(self.buckets[b]) >= batch_size]
        
        if available_buckets:
            # Randomly select a bucket
            selected_bucket = random.choice(available_buckets)
            # Randomly sample batch_size items
            items = random.sample(self.buckets[selected_bucket], batch_size)
            batch_buckets[selected_bucket] = items
        
        return batch_buckets


class SDXLDataset(Dataset):
    """Dataset for SDXL training with tag files."""
    
    def __init__(
        self,
        data_root: str,
        tokenizer_l: CLIPTokenizer,
        tokenizer_g: CLIPTokenizer,
        resolution: int = 1024,
        center_crop: bool = True,
        random_flip: bool = True,
        use_aspect_ratio_bucket: bool = True,
        cache_latents: bool = False,
        vae: Optional[torch.nn.Module] = None,
    ):
        self.data_root = Path(data_root)
        self.tokenizer_l = tokenizer_l
        self.tokenizer_g = tokenizer_g
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.use_aspect_ratio_bucket = use_aspect_ratio_bucket
        self.cache_latents = cache_latents
        self.vae = vae
        
        # Load metadata
        self.image_paths = []
        self.tag_data = []
        self.metadata = []
        
        # Initialize aspect ratio buckets
        self.bucket_manager = AspectRatioBucket(base_resolution=resolution)
        
        # Load dataset
        self._load_dataset()
        
        # Cache for latents
        self.latent_cache = {} if cache_latents else None
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
        ])
    
    def _load_dataset(self):
        """Load image paths and associated tag data."""
        # Supported image formats
        image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
        
        # Look for images and their corresponding tag files
        all_images = []
        for ext in image_extensions:
            all_images.extend(self.data_root.glob(f"**/{ext}"))
        
        for image_path in sorted(all_images):
            tag_path = image_path.with_suffix(".txt")
            json_path = image_path.with_suffix(".json")
            
            if tag_path.exists():
                # Read tags
                with open(tag_path, 'r', encoding='utf-8') as f:
                    tags = f.read().strip()
                
                # Read metadata if exists
                metadata = {}
                if json_path.exists():
                    with open(json_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                
                # Get image dimensions
                with Image.open(image_path) as img:
                    width, height = img.size
                
                # Store data
                idx = len(self.image_paths)
                self.image_paths.append(str(image_path))
                self.tag_data.append(tags)
                self.metadata.append(metadata)
                
                # Add to bucket
                if self.use_aspect_ratio_bucket:
                    self.bucket_manager.add_item(idx, width, height)
        
        print(f"Loaded {len(self.image_paths)} images from {self.data_root}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def _load_and_preprocess_image(self, image_path: str, target_size: Tuple[int, int]) -> torch.Tensor:
        """Load and preprocess an image."""
        img = Image.open(image_path).convert('RGB')
        
        # Resize and crop
        if self.center_crop:
            # Center crop to target aspect ratio
            img_w, img_h = img.size
            target_w, target_h = target_size
            target_aspect = target_w / target_h
            img_aspect = img_w / img_h
            
            if img_aspect > target_aspect:
                # Image is wider, crop width
                new_width = int(img_h * target_aspect)
                left = (img_w - new_width) // 2
                img = img.crop((left, 0, left + new_width, img_h))
            else:
                # Image is taller, crop height
                new_height = int(img_w / target_aspect)
                top = (img_h - new_height) // 2
                img = img.crop((0, top, img_w, top + new_height))
        
        # Resize to target size
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Random horizontal flip
        if self.random_flip and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Convert to tensor
        img_tensor = self.transform(img)
        
        return img_tensor
    
    def _tokenize_caption(self, caption: str) -> Dict[str, torch.Tensor]:
        """Tokenize caption for both CLIP models."""
        # Tokenize for CLIP-L
        tokens_l = self.tokenizer_l(
            caption,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize for CLIP-G
        tokens_g = self.tokenizer_g(
            caption,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids_l": tokens_l["input_ids"].squeeze(0),
            "attention_mask_l": tokens_l["attention_mask"].squeeze(0),
            "input_ids_g": tokens_g["input_ids"].squeeze(0),
            "attention_mask_g": tokens_g["attention_mask"].squeeze(0),
        }
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset."""
        # Get paths and data
        image_path = self.image_paths[idx]
        caption = self.tag_data[idx]
        metadata = self.metadata[idx]
        
        # Determine target size
        if self.use_aspect_ratio_bucket:
            with Image.open(image_path) as img:
                width, height = img.size
            target_size = self.bucket_manager.get_bucket_resolution(width, height)
        else:
            target_size = (self.resolution, self.resolution)
        
        # Load and preprocess image
        image = self._load_and_preprocess_image(image_path, target_size)
        
        # Encode to latent if VAE is provided and caching is enabled
        if self.vae is not None and self.cache_latents:
            cache_key = f"{image_path}_{target_size}"
            if cache_key in self.latent_cache:
                latent = self.latent_cache[cache_key]
            else:
                with torch.no_grad():
                    latent = self.vae.encode(image.unsqueeze(0)).latent_dist.sample()
                    latent = latent.squeeze(0) * 0.18215  # SDXL VAE scaling factor
                    if self.cache_latents:
                        self.latent_cache[cache_key] = latent
        else:
            latent = None
        
        # Tokenize caption
        tokens = self._tokenize_caption(caption)
        
        return {
            "image": image,
            "latent": latent if latent is not None else image,
            "caption": caption,
            "target_size": torch.tensor(target_size),
            **tokens,
            "metadata": metadata,
        }


class BucketBatchSampler:
    """Custom batch sampler for aspect ratio bucketing."""
    
    def __init__(
        self,
        dataset: SDXLDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.bucket_manager = dataset.bucket_manager
    
    def __iter__(self):
        # Create batches from buckets
        batches = []
        
        for bucket, indices in self.bucket_manager.buckets.items():
            # Shuffle indices within bucket if needed
            if self.shuffle:
                indices = indices.copy()
                random.shuffle(indices)
            
            # Create batches
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        
        # Shuffle batches
        if self.shuffle:
            random.shuffle(batches)
        
        for batch in batches:
            yield batch
    
    def __len__(self):
        total = 0
        for bucket, indices in self.bucket_manager.buckets.items():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += (len(indices) + self.batch_size - 1) // self.batch_size
        return total


def create_dataloader(
    data_root: str,
    tokenizer_l: CLIPTokenizer,
    tokenizer_g: CLIPTokenizer,
    batch_size: int = 1,
    resolution: int = 1024,
    num_workers: int = 4,
    use_aspect_ratio_bucket: bool = True,
    **kwargs
) -> DataLoader:
    """Create a DataLoader with aspect ratio bucketing."""
    dataset = SDXLDataset(
        data_root=data_root,
        tokenizer_l=tokenizer_l,
        tokenizer_g=tokenizer_g,
        resolution=resolution,
        use_aspect_ratio_bucket=use_aspect_ratio_bucket,
        **kwargs
    )
    
    if use_aspect_ratio_bucket:
        batch_sampler = BucketBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
    
    return dataloader


if __name__ == "__main__":
    # Test dataset loading
    from transformers import CLIPTokenizer
    
    # Initialize tokenizers
    tokenizer_l = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer_g = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
    
    # Create dataset
    dataset = SDXLDataset(
        data_root="./data/images",
        tokenizer_l=tokenizer_l,
        tokenizer_g=tokenizer_g,
        resolution=1024,
        use_aspect_ratio_bucket=True,
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of buckets: {len(dataset.bucket_manager.buckets)}")
    
    # Test loading an item
    if len(dataset) > 0:
        item = dataset[0]
        print(f"Sample item keys: {item.keys()}")
        print(f"Image shape: {item['image'].shape}")
        print(f"Caption: {item['caption'][:100]}...")

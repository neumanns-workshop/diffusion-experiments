"""
Training utilities for GANs.

This module provides functions for training GANs on geometric datasets.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from typing import Tuple, List, Dict, Any, Optional

from art_generator.models.gan import DCGAN

class GeometricDataset(Dataset):
    """
    Dataset class for loading geometric images.
    """
    
    def __init__(
        self,
        data_dir: str,
        image_size: int = 64,
        channels: int = 1  # 1 for grayscale, 3 for RGB
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing the images
            image_size: Size to resize the images to
            channels: Number of channels (1 for grayscale, 3 for RGB)
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.channels = channels
        
        # Get all image files
        self.image_files = []
        for file in os.listdir(data_dir):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                self.image_files.append(os.path.join(data_dir, file))
        
        # Set up transformations
        if channels == 1:
            # Grayscale
            self.transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
            ])
        else:
            # RGB
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
            ])
    
    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get an image from the dataset.
        
        Args:
            idx: Index of the image
            
        Returns:
            Tensor of shape (channels, image_size, image_size)
        """
        # Load the image
        image_path = self.image_files[idx]
        image = Image.open(image_path)
        
        # Convert RGBA to RGB if needed
        if image.mode == 'RGBA' and self.channels == 3:
            # Create a white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            # Paste the image on the background using the alpha channel
            background.paste(image, mask=image.split()[3])
            image = background
        elif image.mode != 'L' and self.channels == 1:
            # Force grayscale for non-grayscale images when channels=1
            image = image.convert('L')
        elif image.mode != 'RGB' and self.channels == 3:
            # Force RGB for non-RGB images when channels=3
            image = image.convert('RGB')
        
        # Apply transformations
        return self.transform(image)


def train_gan(
    data_dir: str,
    output_dir: str,
    batch_size: int = 64,
    epochs: int = 100,
    latent_dim: int = 100,
    channels: int = 1,  # 1 for grayscale, 3 for RGB
    image_size: int = 64,
    learning_rate: float = 0.0002,
    beta1: float = 0.5,
    save_interval: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    sample_size: int = 16,  # Number of images to sample during training
    noise_scale: float = 1.0,  # Scale of the noise for latent vector
    use_multiple_noise_samples: bool = True  # Use multiple noise samples for each real image
) -> DCGAN:
    """
    Train a GAN on a dataset of geometric images.
    
    Args:
        data_dir: Directory containing the images
        output_dir: Directory to save the models and samples
        batch_size: Batch size for training
        epochs: Number of epochs to train for
        latent_dim: Dimension of the latent (noise) vector
        channels: Number of channels in the images (1 for grayscale, 3 for RGB)
        image_size: Size of the images (assumes square)
        learning_rate: Learning rate for optimizers
        beta1: Beta1 hyperparameter for Adam optimizers
        save_interval: Interval (in epochs) to save models and samples
        device: Device to use for training ('cuda' or 'cpu')
        sample_size: Number of images to sample during training
        noise_scale: Standard deviation of noise for the latent vector
        use_multiple_noise_samples: Generate multiple samples per real image to prevent mode collapse
        
    Returns:
        Trained DCGAN model
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, "models")
    samples_dir = os.path.join(output_dir, "samples")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    
    # Create dataset and dataloader
    dataset = GeometricDataset(data_dir, image_size, channels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize the GAN
    gan = DCGAN(
        latent_dim=latent_dim,
        channels=channels,
        feature_maps=64,
        image_size=image_size,
        learning_rate=learning_rate,
        beta1=beta1,
        device=device
    )
    
    # Fixed noise for sampling
    fixed_noise = torch.randn(sample_size, latent_dim, device=device) * noise_scale
    
    # Track losses
    losses = {
        "g_loss": [],
        "d_loss": [],
        "feature_loss": []
    }
    
    # Training loop
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_losses = {
            "g_loss": [],
            "d_loss": [],
            "feature_loss": []
        }
        
        # Train on batches
        with tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}") as pbar:
            for batch_imgs in pbar:
                batch_size = batch_imgs.size(0)
                
                # Use multiple noise samples for variety (helps prevent mode collapse)
                if use_multiple_noise_samples:
                    # Train with 2-3 different noise vectors for each real batch
                    num_samples = np.random.randint(2, 4)
                    for _ in range(num_samples):
                        # Generate new noise for each training step
                        noise = torch.randn(batch_size, latent_dim, device=device) * noise_scale
                        # Train on batch with this noise
                        batch_losses = gan.train_batch(batch_imgs)
                        
                        # Update losses
                        for k, v in batch_losses.items():
                            if k in epoch_losses:
                                epoch_losses[k].append(v)
                else:
                    # Standard training with single noise vector
                    batch_losses = gan.train_batch(batch_imgs)
                    
                    # Update losses
                    for k, v in batch_losses.items():
                        if k in epoch_losses:
                            epoch_losses[k].append(v)
                
                # Update progress bar
                g_loss = epoch_losses["g_loss"][-1] if epoch_losses["g_loss"] else 0
                d_loss = epoch_losses["d_loss"][-1] if epoch_losses["d_loss"] else 0
                feature_loss = epoch_losses["feature_loss"][-1] if epoch_losses["feature_loss"] else 0
                
                pbar.set_postfix({
                    "g_loss": g_loss,
                    "d_loss": d_loss,
                    "feature_loss": feature_loss
                })
        
        # Calculate average losses for the epoch
        for k, v in epoch_losses.items():
            if v:  # Only if there are values
                avg_loss = sum(v) / len(v)
                losses[k].append(avg_loss)
            else:
                losses[k].append(0.0)
        
        # Print epoch stats
        print(f"Epoch {epoch}/{epochs} - g_loss: {losses['g_loss'][-1]:.4f}, d_loss: {losses['d_loss'][-1]:.4f}, feature_loss: {losses.get('feature_loss', [0])[-1]:.4f}")
        
        # Save model and samples at specified intervals
        if epoch % save_interval == 0 or epoch == epochs:
            # Save model
            model_path = os.path.join(model_dir, f"gan_epoch_{epoch}.pth")
            gan.save(model_path)
            
            # Generate and save samples
            samples = gan.generate(sample_size, fixed_noise)
            samples = samples.detach().cpu()
            
            # Create a grid of images
            fig, axes = plt.subplots(int(np.sqrt(sample_size)), int(np.sqrt(sample_size)), figsize=(10, 10))
            axes = axes.flatten()
            
            for i, img in enumerate(samples):
                # Denormalize image
                img = img * 0.5 + 0.5
                
                # Convert to numpy and transpose if needed
                if channels == 1:
                    img = img.squeeze().numpy()
                else:
                    img = img.numpy().transpose(1, 2, 0)  # CHW -> HWC
                
                # Plot
                axes[i].imshow(img, cmap="gray" if channels == 1 else None)
                axes[i].axis("off")
            
            plt.tight_layout()
            plt.savefig(os.path.join(samples_dir, f"samples_epoch_{epoch}.png"))
            plt.close()
            
            # Save some random samples too (not just the fixed noise)
            random_noise = torch.randn(sample_size, latent_dim, device=device) * noise_scale
            random_samples = gan.generate(sample_size, random_noise)
            random_samples = random_samples.detach().cpu()
            
            # Create a grid of images for random samples
            fig, axes = plt.subplots(int(np.sqrt(sample_size)), int(np.sqrt(sample_size)), figsize=(10, 10))
            axes = axes.flatten()
            
            for i, img in enumerate(random_samples):
                # Denormalize image
                img = img * 0.5 + 0.5
                
                # Convert to numpy and transpose if needed
                if channels == 1:
                    img = img.squeeze().numpy()
                else:
                    img = img.numpy().transpose(1, 2, 0)  # CHW -> HWC
                
                # Plot
                axes[i].imshow(img, cmap="gray" if channels == 1 else None)
                axes[i].axis("off")
            
            plt.tight_layout()
            plt.savefig(os.path.join(samples_dir, f"samples_random_epoch_{epoch}.png"))
            plt.close()
    
    # Training complete
    elapsed_time = time.time() - start_time
    print(f"Training complete in {elapsed_time:.2f} seconds")
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(losses["g_loss"], label="Generator Loss")
    plt.plot(losses["d_loss"], label="Discriminator Loss")
    if "feature_loss" in losses and any(losses["feature_loss"]):
        plt.plot(losses["feature_loss"], label="Feature Loss")
    plt.legend()
    plt.title("Training Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "losses.png"))
    plt.close()
    
    # Save final model directly in output directory
    final_model_path = os.path.join(output_dir, "final_model.pth")
    gan.save(final_model_path)
    
    return gan 
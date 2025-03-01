"""
GAN model definitions for generating geometric art.

This module contains the Generator, Discriminator, and DCGAN classes.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Optional, Union

class Generator(nn.Module):
    """
    Generator network for the GAN.
    
    Transforms a latent vector into an image.
    """
    
    def __init__(
        self,
        latent_dim: int = 100,
        channels: int = 1,
        feature_maps: int = 64,
        image_size: int = 64
    ):
        """
        Initialize the Generator.
        
        Args:
            latent_dim: Dimension of the latent (noise) vector
            channels: Number of output channels (1 for grayscale, 3 for RGB)
            feature_maps: Base size of feature maps
            image_size: Size of output images (assumes square)
        """
        super(Generator, self).__init__()
        
        # Calculate number of upsampling layers based on image size
        self.num_layers = int(torch.log2(torch.tensor(image_size))) - 2
        self.latent_dim = latent_dim
        self.channels = channels
        self.feature_maps = feature_maps
        
        # Input: latent_dim x 1 x 1
        layers = []
        
        # First layer: latent_dim -> feature_maps*8 x 4 x 4
        layers.append(
            nn.ConvTranspose2d(
                latent_dim,
                feature_maps * 8,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            )
        )
        layers.append(nn.BatchNorm2d(feature_maps * 8))
        layers.append(nn.ReLU(True))
        
        # Middle layers: Upsampling
        size_mult = 8
        for i in range(self.num_layers - 1):
            new_size_mult = size_mult // 2
            layers.append(
                nn.ConvTranspose2d(
                    feature_maps * size_mult,
                    feature_maps * new_size_mult,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                )
            )
            layers.append(nn.BatchNorm2d(feature_maps * new_size_mult))
            layers.append(nn.ReLU(True))
            size_mult = new_size_mult
        
        # Final layer: feature_maps -> channels
        layers.append(
            nn.ConvTranspose2d(
                feature_maps * size_mult,
                channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            )
        )
        layers.append(nn.Tanh())  # Output range: [-1, 1]
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Generator.
        
        Args:
            z: Latent (noise) vector of shape (batch_size, latent_dim)
            
        Returns:
            Generated images of shape (batch_size, channels, image_size, image_size)
        """
        # Reshape z to (batch_size, latent_dim, 1, 1)
        z = z.unsqueeze(-1).unsqueeze(-1)
        return self.model(z)


class Discriminator(nn.Module):
    """
    Discriminator network for the GAN.
    
    Classifies images as real or fake.
    """
    
    def __init__(
        self,
        channels: int = 1,
        feature_maps: int = 64,
        image_size: int = 64,
        use_spectral_norm: bool = True
    ):
        """
        Initialize the Discriminator.
        
        Args:
            channels: Number of input channels (1 for grayscale, 3 for RGB)
            feature_maps: Base size of feature maps
            image_size: Size of input images (assumes square)
            use_spectral_norm: Whether to use spectral normalization for stability
        """
        super(Discriminator, self).__init__()
        
        # Calculate number of downsampling layers based on image size
        self.num_layers = int(torch.log2(torch.tensor(image_size))) - 2
        self.channels = channels
        self.feature_maps = feature_maps
        
        # Create normalization function based on parameter
        norm_layer = nn.utils.spectral_norm if use_spectral_norm else lambda x: x
        
        # Input: channels x image_size x image_size
        layers = []
        
        # First layer: channels -> feature_maps
        layers.append(
            norm_layer(nn.Conv2d(
                channels,
                feature_maps,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ))
        )
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Middle layers: Downsampling
        size_mult = 1
        for i in range(self.num_layers - 1):
            new_size_mult = size_mult * 2
            layers.append(
                norm_layer(nn.Conv2d(
                    feature_maps * size_mult,
                    feature_maps * new_size_mult,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                ))
            )
            # Use instance norm instead of batch norm for discriminator
            layers.append(nn.InstanceNorm2d(feature_maps * new_size_mult, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            size_mult = new_size_mult
        
        # Final layer: feature_maps*size_mult -> 1
        layers.append(
            norm_layer(nn.Conv2d(
                feature_maps * size_mult,
                1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ))
        )
        # No sigmoid here - we'll use BCEWithLogitsLoss
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Discriminator.
        
        Args:
            x: Images of shape (batch_size, channels, image_size, image_size)
            
        Returns:
            Classification scores of shape (batch_size, 1, 1, 1)
        """
        return self.model(x)


class DCGAN:
    """
    Deep Convolutional GAN model.
    
    Combines Generator and Discriminator networks.
    """
    
    def __init__(
        self,
        latent_dim: int = 100,
        channels: int = 1,
        feature_maps: int = 64,
        image_size: int = 64,
        learning_rate: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the DCGAN.
        
        Args:
            latent_dim: Dimension of the latent (noise) vector
            channels: Number of channels in images (1 for grayscale, 3 for RGB)
            feature_maps: Base size of feature maps
            image_size: Size of images (assumes square)
            learning_rate: Learning rate for optimizers
            beta1: Beta1 hyperparameter for Adam optimizers
            beta2: Beta2 hyperparameter for Adam optimizers
            device: Device to use for training ('cuda' or 'cpu')
        """
        self.latent_dim = latent_dim
        self.channels = channels
        self.feature_maps = feature_maps
        self.image_size = image_size
        self.device = device
        
        # Initialize networks
        self.generator = Generator(
            latent_dim=latent_dim,
            channels=channels,
            feature_maps=feature_maps,
            image_size=image_size
        ).to(device)
        
        self.discriminator = Discriminator(
            channels=channels,
            feature_maps=feature_maps,
            image_size=image_size
        ).to(device)
        
        # Initialize optimizers
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2)
        )
        
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2)
        )
        
        # Loss function (BCE with logits for improved numerical stability)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Labels for real and fake images
        self.real_label = 1
        self.fake_label = 0
    
    def train_batch(self, real_images: torch.Tensor) -> Dict[str, float]:
        """
        Train the GAN on a batch of images.
        
        Args:
            real_images: Batch of real images
            
        Returns:
            Dictionary of losses
        """
        batch_size = real_images.size(0)
        real_images = real_images.to(self.device)
        
        # Labels
        real_labels = torch.full((batch_size, 1, 1, 1), self.real_label, 
                                device=self.device, dtype=torch.float32)
        fake_labels = torch.full((batch_size, 1, 1, 1), self.fake_label, 
                                device=self.device, dtype=torch.float32)
        
        # Add noise to labels for smoother training
        real_labels = real_labels * 0.9 + 0.1 * torch.rand_like(real_labels)
        fake_labels = fake_labels * 0.1 + 0.0 * torch.rand_like(fake_labels)
        
        # -----------------
        # Train Discriminator
        # -----------------
        self.optimizer_d.zero_grad()
        
        # Real images
        real_output = self.discriminator(real_images)
        d_loss_real = self.criterion(real_output, real_labels)
        
        # Fake images
        noise = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_images = self.generator(noise)
        fake_output = self.discriminator(fake_images.detach())  # Detach to avoid training G
        d_loss_fake = self.criterion(fake_output, fake_labels)
        
        # Combined loss and gradient update
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.optimizer_d.step()
        
        # -----------------
        # Train Generator
        # -----------------
        self.optimizer_g.zero_grad()
        
        # Try to fool the discriminator
        output = self.discriminator(fake_images)
        g_loss = self.criterion(output, real_labels)
        
        # Add feature matching loss to reduce mode collapse
        # This encourages the generator to produce images that match the statistics
        # of real images at the feature level
        fake_features = self.discriminator.model[:-1](fake_images)
        real_features = self.discriminator.model[:-1](real_images).detach()
        feature_loss = nn.functional.mse_loss(fake_features, real_features) * 0.1
        
        # Combined loss and gradient update
        g_total_loss = g_loss + feature_loss
        g_total_loss.backward()
        self.optimizer_g.step()
        
        return {
            "d_loss": d_loss.item(),
            "g_loss": g_loss.item(),
            "feature_loss": feature_loss.item()
        }
    
    def generate(
        self, 
        num_images: int = 1, 
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate images using the trained generator.
        
        Args:
            num_images: Number of images to generate
            noise: Optional pre-defined noise vectors
            
        Returns:
            Generated images
        """
        self.generator.eval()
        with torch.no_grad():
            # Generate noise if not provided
            if noise is None:
                noise = torch.randn(num_images, self.latent_dim, device=self.device)
            
            # Generate images
            images = self.generator(noise)
        
        self.generator.train()
        return images
    
    def save(self, path: str) -> None:
        """
        Save the model to the specified path.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizer_g_state_dict": self.optimizer_g.state_dict(),
            "optimizer_d_state_dict": self.optimizer_d.state_dict(),
            "latent_dim": self.latent_dim,
            "channels": self.channels,
            "feature_maps": self.feature_maps,
            "image_size": self.image_size
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> 'DCGAN':
        """
        Load a model from the specified path.
        
        Args:
            path: Path to load the model from
            device: Device to load the model to
            
        Returns:
            Loaded DCGAN model
        """
        checkpoint = torch.load(path, map_location=device)
        
        # Initialize the model
        model = cls(
            latent_dim=checkpoint["latent_dim"],
            channels=checkpoint["channels"],
            feature_maps=checkpoint["feature_maps"],
            image_size=checkpoint["image_size"],
            device=device
        )
        
        # Load state dictionaries
        model.generator.load_state_dict(checkpoint["generator_state_dict"])
        model.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        model.optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
        model.optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
        
        return model 
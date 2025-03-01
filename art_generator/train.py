"""
Train GAN models on geometric datasets.
"""

import os
import argparse
import torch
from typing import Optional

from art_generator.models.training import train_gan

def main():
    """Main function for training GAN models."""
    parser = argparse.ArgumentParser(description="Train a GAN model on a geometric dataset")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Directory containing the dataset images")
    parser.add_argument("--output", type=str, required=True,
                        help="Directory to save the trained model and samples")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs to train")
    parser.add_argument("--latent-dim", type=int, default=100,
                        help="Dimension of the latent space")
    parser.add_argument("--channels", type=int, default=1,
                        help="Number of channels in the images (1 for grayscale, 3 for RGB)")
    parser.add_argument("--image-size", type=int, default=64,
                        help="Size of the images (width and height)")
    parser.add_argument("--learning-rate", type=float, default=0.0002,
                        help="Learning rate for optimizers")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="Interval (in epochs) to save models and samples")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training ('cuda' or 'cpu')")
    
    args = parser.parse_args()
    
    # Train the GAN
    gan = train_gan(
        data_dir=args.dataset,
        output_dir=args.output,
        batch_size=args.batch_size,
        epochs=args.epochs,
        latent_dim=args.latent_dim,
        channels=args.channels,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        save_interval=args.save_interval,
        device=args.device
    )
    
    # Save the final model
    model_path = os.path.join(args.output, "final_model.pth")
    gan.save(model_path)
    print(f"Final model saved to {model_path}")

if __name__ == "__main__":
    main() 
"""
Demo script to showcase the full geometric art generation pipeline.

This script demonstrates:
1. Dataset generation with various styles
2. Model training on the generated datasets
3. Image generation using the trained models
"""

import os
import argparse
import shutil
from typing import List

from art_generator.datasets.generate import generate_dataset
from art_generator.models.training import train_gan
from art_generator.generate import generate_images
from art_generator.utils.config import load_config, PREDEFINED_CONFIGS

# Default demo settings
DEFAULT_BASE_DIR = "demo_output"
DEFAULT_STYLES = ["grid", "triangles", "circles", "brutalist"]
DEFAULT_COUNT = 100
DEFAULT_EPOCHS = 50


def run_demo(
    base_dir: str = DEFAULT_BASE_DIR,
    styles: List[str] = DEFAULT_STYLES,
    dataset_count: int = DEFAULT_COUNT,
    epochs: int = DEFAULT_EPOCHS,
    clean_output: bool = False,
    skip_training: bool = False,
    generate_only: bool = False,
    config_path: str = None
):
    """
    Run the full pipeline demo.
    
    Args:
        base_dir: Base directory for output
        styles: List of styles to generate
        dataset_count: Number of images to generate per style
        epochs: Number of epochs to train each model
        clean_output: If True, clean output directory before running
        skip_training: If True, skip training and use existing models
        generate_only: If True, only generate images from existing models
        config_path: Path to a config file to use instead of built-in styles
    """
    # Create base directory
    os.makedirs(base_dir, exist_ok=True)
    
    # Clean output if requested
    if clean_output and os.path.exists(base_dir):
        shutil.rmtree(base_dir)
        os.makedirs(base_dir, exist_ok=True)
    
    if config_path:
        # Use custom config
        config = load_config(config_path)
        styles = [config.style]
        
        # Create directory structure
        dataset_dir = os.path.join(base_dir, "datasets", config.name)
        model_dir = os.path.join(base_dir, "models", config.name)
        output_dir = os.path.join(base_dir, "generated", config.name)
        
        os.makedirs(dataset_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate dataset if not in generate_only mode
        if not generate_only:
            print(f"\n=== Generating {config.name} dataset ===")
            generate_dataset(
                style=config.style,
                count=dataset_count,
                output_dir=dataset_dir,
                seed=42,
                image_size=(config.image_size, config.image_size),
                color_palette=config.color_palette,
                **config.get_style_params()
            )
        
        # Train model if not in generate_only or skip_training mode
        if not generate_only and not skip_training:
            print(f"\n=== Training {config.name} model ===")
            train_gan(
                data_dir=dataset_dir,
                output_dir=model_dir,
                epochs=epochs,
                channels=config.channels,
                image_size=64,  # Use smaller image size for faster training
                batch_size=32,
                save_interval=5
            )
        
        # Generate images
        print(f"\n=== Generating images from {config.name} model ===")
        # Look for the final model in the main model directory
        model_path = os.path.join(model_dir, "final_model.pth")
        
        if not os.path.exists(model_path):
            # Try to find the latest epoch model
            model_files = []
            models_subdir = os.path.join(model_dir, "models")
            # Check both the model_dir and the models subdirectory
            for directory in [model_dir, models_subdir]:
                if os.path.exists(directory):
                    model_files.extend([
                        os.path.join(directory, f) for f in os.listdir(directory) 
                        if f.endswith(".pth")
                    ])
            
            if model_files:
                # Sort by modification time to get the latest
                model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                model_path = model_files[0]
                print(f"Using most recent model: {model_path}")
            else:
                print(f"No model found in {model_dir}, skipping image generation.")
                return
        
        try:
            generate_images(
                model_path=model_path,
                output_dir=output_dir,
                count=16,
                grid_output=os.path.join(output_dir, "grid.png"),
                noise_seed=42
            )
            print(f"Generated 16 images in {output_dir}")
        except Exception as e:
            print(f"Error generating images: {e}")
    else:
        # Use built-in styles
        for style in styles:
            # Create directory structure
            dataset_dir = os.path.join(base_dir, "datasets", style)
            model_dir = os.path.join(base_dir, "models", style)
            output_dir = os.path.join(base_dir, "generated", style)
            
            os.makedirs(dataset_dir, exist_ok=True)
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate dataset if not in generate_only mode
            if not generate_only:
                print(f"\n=== Generating {style} dataset ===")
                generate_dataset(
                    style=style,
                    count=dataset_count,
                    output_dir=dataset_dir,
                    seed=42
                )
            
            # Train model if not in generate_only or skip_training mode
            if not generate_only and not skip_training:
                print(f"\n=== Training {style} model ===")
                train_gan(
                    data_dir=dataset_dir,
                    output_dir=model_dir,
                    epochs=epochs,
                    channels=1,  # Use grayscale for faster training
                    image_size=64,  # Use smaller image size for faster training
                    batch_size=32,
                    save_interval=5
                )
            
            # Generate images
            print(f"\n=== Generating images from {style} model ===")
            # Look for the final model in the main model directory
            model_path = os.path.join(model_dir, "final_model.pth")
            
            if not os.path.exists(model_path):
                # Try to find the latest epoch model
                model_files = []
                models_subdir = os.path.join(model_dir, "models")
                # Check both the model_dir and the models subdirectory
                for directory in [model_dir, models_subdir]:
                    if os.path.exists(directory):
                        model_files.extend([
                            os.path.join(directory, f) for f in os.listdir(directory) 
                            if f.endswith(".pth")
                        ])
                
                if model_files:
                    # Sort by modification time to get the latest
                    model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                    model_path = model_files[0]
                    print(f"Using most recent model: {model_path}")
                else:
                    print(f"No model found in {model_dir}, skipping image generation.")
                    continue
            
            try:
                generate_images(
                    model_path=model_path,
                    output_dir=output_dir,
                    count=16,
                    grid_output=os.path.join(output_dir, "grid.png"),
                    noise_seed=42
                )
                print(f"Generated 16 images in {output_dir}")
            except Exception as e:
                print(f"Error generating images: {e}")


def main():
    """Main function for the demo script."""
    parser = argparse.ArgumentParser(description="Run the geometric art generation pipeline demo")
    parser.add_argument("--output", type=str, default=DEFAULT_BASE_DIR,
                        help=f"Base directory for output (default: {DEFAULT_BASE_DIR})")
    parser.add_argument("--styles", type=str, nargs="+", default=DEFAULT_STYLES,
                        help=f"Styles to generate (default: {' '.join(DEFAULT_STYLES)})")
    parser.add_argument("--dataset-count", type=int, default=DEFAULT_COUNT,
                        help=f"Number of images to generate per style (default: {DEFAULT_COUNT})")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Number of epochs to train each model (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--clean", action="store_true",
                        help="Clean output directory before running")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training and use existing models")
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate images from existing models")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a config file to use instead of built-in styles")
    
    args = parser.parse_args()
    
    # Run the demo
    run_demo(
        base_dir=args.output,
        styles=args.styles,
        dataset_count=args.dataset_count,
        epochs=args.epochs,
        clean_output=args.clean,
        skip_training=args.skip_training,
        generate_only=args.generate_only,
        config_path=args.config
    )
    
    print("\n=== Demo complete! ===")
    print(f"Generated datasets are in {os.path.join(args.output, 'datasets')}")
    print(f"Trained models are in {os.path.join(args.output, 'models')}")
    print(f"Generated images are in {os.path.join(args.output, 'generated')}")


if __name__ == "__main__":
    main() 
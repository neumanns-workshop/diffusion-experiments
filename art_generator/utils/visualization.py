"""
Visualization utilities for displaying and saving generated images.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Optional, Union, Tuple

def plot_images(
    images: Union[List[np.ndarray], np.ndarray, torch.Tensor],
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 10),
    cmap: Optional[str] = "gray",
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a grid of images.
    
    Args:
        images: List of images or tensor of shape (N, C, H, W) or (N, H, W)
        rows: Number of rows in the grid (inferred if None)
        cols: Number of columns in the grid (inferred if None)
        figsize: Figure size (width, height)
        cmap: Colormap for grayscale images
        title: Title for the figure
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    # Convert tensor to numpy if needed
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
        
        # Handle different tensor formats
        if images.ndim == 4:
            # (N, C, H, W) -> (N, H, W, C)
            images = images.transpose(0, 2, 3, 1)
            
            # If single channel, squeeze
            if images.shape[-1] == 1:
                images = images.squeeze(-1)
    
    # Handle single image
    if isinstance(images, np.ndarray) and images.ndim in [2, 3]:
        images = [images]
    
    # Determine grid size
    n = len(images)
    if rows is None and cols is None:
        # Make a square grid
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
    elif rows is None:
        # Calculate rows from cols
        rows = int(np.ceil(n / cols))
    elif cols is None:
        # Calculate cols from rows
        cols = int(np.ceil(n / rows))
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Handle single row/column case
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot images
    for i, ax in enumerate(axes):
        if i < n:
            image = images[i]
            # Determine if color or grayscale
            if image.ndim == 3 and image.shape[2] in [3, 4]:
                ax.imshow(image)
            else:
                ax.imshow(image, cmap=cmap)
            ax.axis("off")
        else:
            # Hide empty subplots
            ax.axis("off")
    
    # Set title
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    # Save figure
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    
    return fig

def save_image_grid(
    images: Union[List[np.ndarray], np.ndarray, torch.Tensor],
    output_path: str,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 10),
    cmap: Optional[str] = "gray",
    title: Optional[str] = None
) -> None:
    """
    Save a grid of images to a file.
    
    Args:
        images: List of images or tensor of shape (N, C, H, W) or (N, H, W)
        output_path: Path to save the grid
        rows: Number of rows in the grid (inferred if None)
        cols: Number of columns in the grid (inferred if None)
        figsize: Figure size (width, height)
        cmap: Colormap for grayscale images
        title: Title for the figure
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Plot and save
    fig = plot_images(
        images=images,
        rows=rows,
        cols=cols,
        figsize=figsize,
        cmap=cmap,
        title=title,
        save_path=output_path
    )
    
    # Close the figure
    plt.close(fig) 
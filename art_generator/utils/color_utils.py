"""
Utility functions for color handling in geometric art generation.
"""

from typing import Union, List, Tuple, Any

def normalize_color(color: Union[str, List, Tuple]) -> Union[str, Tuple[float, ...]]:
    """
    Normalize color to format accepted by matplotlib (0-1 range for RGB values).
    
    Args:
        color: Color as string (name or hex) or tuple/list of RGB(A) values
        
    Returns:
        Normalized color suitable for matplotlib
    """
    # If it's a string (name or hex), return as is
    if isinstance(color, str):
        return color
        
    # If it's a sequence of numbers
    if isinstance(color, (list, tuple)):
        # Check if values are in 0-255 range
        if any(isinstance(v, int) and v > 1 for v in color):
            # Convert from 0-255 to 0-1
            return tuple(float(v) / 255.0 for v in color)
        # Otherwise, assume already in 0-1 range
        return tuple(float(v) for v in color)
        
    # If we don't recognize the format, return as is
    return color 
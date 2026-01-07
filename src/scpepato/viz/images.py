"""Image loading and processing utilities for cell visualization.

This module provides functions for loading cell images from brieflow TIFF files,
extracting cell crops, and creating composite images for visualization.
"""

import base64
import io
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import tifffile
except ImportError:
    tifffile = None


def get_image_path(
    plate: int | str | float,
    well: str,
    tile: int | str | float,
    images_dir: str | Path,
) -> Path:
    """Construct the path to a tile image file.

    Args:
        plate: Plate number
        well: Well identifier (e.g., 'A1')
        tile: Tile number
        images_dir: Root directory containing images

    Returns:
        Path to the aligned TIFF file
    """
    images_dir = Path(images_dir)
    # Convert to int to handle float values from parquet
    plate = int(plate)
    tile = int(tile)
    filename = f"P-{plate}_W-{well}_T-{tile}__aligned.tiff"
    return images_dir / filename


@lru_cache(maxsize=32)
def load_tile_image(image_path: str | Path) -> np.ndarray:
    """Load a tile image with caching.

    Uses LRU cache to avoid repeated disk reads for cells in the same tile.

    Args:
        image_path: Path to the TIFF file

    Returns:
        Image array with shape (channels, height, width)
    """
    if tifffile is None:
        raise ImportError(
            "tifffile is required for image loading. Install with: pip install tifffile"
        )

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    return tifffile.imread(str(image_path))


def extract_cell_image(
    image: np.ndarray,
    bounds: tuple[int, int, int, int],
    padding: int = 10,
) -> np.ndarray:
    """Extract a cell crop from a tile image.

    Args:
        image: Tile image array (channels, height, width)
        bounds: Bounding box (min_row, min_col, max_row, max_col)
        padding: Padding around the bounding box

    Returns:
        Cropped cell image (channels, crop_height, crop_width)
    """
    i0, j0, i1, j1 = bounds
    i0 -= padding
    j0 -= padding
    i1 += padding
    j1 += padding

    # Handle boundaries
    h, w = image.shape[-2], image.shape[-1]
    i0_clip = max(i0, 0)
    j0_clip = max(j0, 0)
    i1_clip = min(i1, h)
    j1_clip = min(j1, w)

    # Create output with padding (zeros for out-of-bounds)
    out_shape = image.shape[:-2] + (i1 - i0, j1 - j0)
    cell_crop = np.zeros(out_shape, dtype=image.dtype)

    # Fill in the valid region
    out_i0 = i0_clip - i0
    out_j0 = j0_clip - j0
    out_i1 = out_i0 + (i1_clip - i0_clip)
    out_j1 = out_j0 + (j1_clip - j0_clip)

    cell_crop[..., out_i0:out_i1, out_j0:out_j1] = image[..., i0_clip:i1_clip, j0_clip:j1_clip]

    return cell_crop


def normalize_channel(
    channel: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> np.ndarray:
    """Normalize a channel to 0-1 range using percentile clipping.

    Args:
        channel: Single-channel image array
        lower_percentile: Lower percentile for clipping
        upper_percentile: Upper percentile for clipping

    Returns:
        Normalized array in 0-1 range
    """
    vmin = np.percentile(channel, lower_percentile)
    vmax = np.percentile(channel, upper_percentile)

    if vmax <= vmin:
        return np.zeros_like(channel, dtype=np.float32)

    normalized = (channel.astype(np.float32) - vmin) / (vmax - vmin)
    return np.clip(normalized, 0, 1)


def make_composite(
    channels: np.ndarray,
    channel_colors: list[tuple[float, float, float]] | None = None,
) -> np.ndarray:
    """Create an RGB composite from multi-channel image.

    Args:
        channels: Multi-channel image (n_channels, height, width)
        channel_colors: List of RGB tuples (0-1) for each channel.
            Defaults to: blue, green, red, cyan for up to 4 channels.

    Returns:
        RGB composite image (height, width, 3) as uint8
    """
    n_channels = channels.shape[0]

    # Default color scheme
    if channel_colors is None:
        default_colors = [
            (0.0, 0.0, 1.0),  # Blue (DAPI)
            (0.0, 1.0, 0.0),  # Green
            (1.0, 0.0, 0.0),  # Red
            (0.0, 1.0, 1.0),  # Cyan
        ]
        channel_colors = default_colors[:n_channels]

    h, w = channels.shape[1], channels.shape[2]
    composite = np.zeros((h, w, 3), dtype=np.float32)

    for i, (channel, color) in enumerate(zip(channels, channel_colors)):
        # Normalize each channel
        norm_channel = normalize_channel(channel)

        # Add to composite with color
        for c in range(3):
            composite[:, :, c] += norm_channel * color[c]

    # Clip and convert to uint8
    composite = np.clip(composite, 0, 1)
    return (composite * 255).astype(np.uint8)


def encode_image_base64(
    image: np.ndarray,
    format: str = "PNG",
    size: tuple[int, int] | None = None,
) -> str:
    """Encode an image array as base64 string for web display.

    Args:
        image: Image array (height, width, 3) as uint8
        format: Image format (PNG, JPEG, etc.)
        size: Optional resize dimensions (width, height)

    Returns:
        Base64-encoded image string with data URI prefix
    """
    pil_image = Image.fromarray(image)

    if size is not None:
        pil_image = pil_image.resize(size, Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    buffer.seek(0)

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"


def get_cell_composite(
    plate: int | str,
    well: str,
    tile: int | str,
    bounds: tuple[int, int, int, int],
    images_dir: str | Path,
    channel_colors: list[tuple[float, float, float]] | None = None,
    padding: int = 10,
    output_size: tuple[int, int] | None = (100, 100),
) -> str:
    """Get a cell composite image as base64 string.

    Convenience function that combines loading, cropping, and encoding.

    Args:
        plate: Plate number
        well: Well identifier
        tile: Tile number
        bounds: Bounding box (min_row, min_col, max_row, max_col)
        images_dir: Root directory containing images
        channel_colors: RGB colors for each channel
        padding: Padding around bounds
        output_size: Resize dimensions for output (width, height)

    Returns:
        Base64-encoded composite image string
    """
    # Get image path and load
    image_path = get_image_path(plate, well, tile, images_dir)
    tile_image = load_tile_image(str(image_path))

    # Extract cell crop
    cell_crop = extract_cell_image(tile_image, bounds, padding=padding)

    # Create composite
    composite = make_composite(cell_crop, channel_colors=channel_colors)

    # Encode for web
    return encode_image_base64(composite, size=output_size)


def clear_image_cache():
    """Clear the tile image cache."""
    load_tile_image.cache_clear()

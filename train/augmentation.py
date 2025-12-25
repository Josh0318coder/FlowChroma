"""
Reference Image Augmentation for Fusion Training

Only augments reference images, not target frames.
Prevents the model from memorizing reference colors/positions.

Includes:
- Color augmentation (hue, saturation, brightness)
- Spatial augmentation (flip, TPS transform)
"""

import numpy as np
from PIL import Image, ImageEnhance
import random
import torch


def random_hue_shift(image, hue_range=30):
    """
    Random hue shift in HSV color space

    Args:
        image: PIL Image (RGB)
        hue_range: Maximum hue shift in degrees (±hue_range)

    Returns:
        PIL Image with shifted hue
    """
    if hue_range == 0:
        return image

    # Convert to HSV
    image_hsv = image.convert('HSV')
    h, s, v = image_hsv.split()

    # Shift hue (H channel is 0-255 in PIL, representing 0-360 degrees)
    # hue_range degrees = hue_range * 255 / 360
    hue_shift = random.uniform(-hue_range, hue_range) * 255 / 360

    h_array = np.array(h, dtype=np.float32)
    h_array = (h_array + hue_shift) % 256  # Wrap around
    h_shifted = Image.fromarray(h_array.astype(np.uint8), mode='L')

    # Merge back
    image_hsv_shifted = Image.merge('HSV', (h_shifted, s, v))
    return image_hsv_shifted.convert('RGB')


def random_saturation(image, saturation_range=(0.7, 1.3)):
    """
    Random saturation adjustment

    Args:
        image: PIL Image (RGB)
        saturation_range: (min_factor, max_factor)

    Returns:
        PIL Image with adjusted saturation
    """
    if saturation_range == (1.0, 1.0):
        return image

    factor = random.uniform(saturation_range[0], saturation_range[1])
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)


def random_brightness(image, brightness_range=(0.8, 1.2)):
    """
    Random brightness adjustment

    Args:
        image: PIL Image (RGB)
        brightness_range: (min_factor, max_factor)

    Returns:
        PIL Image with adjusted brightness
    """
    if brightness_range == (1.0, 1.0):
        return image

    factor = random.uniform(brightness_range[0], brightness_range[1])
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def random_rgb_flip(image, prob=0.15):
    """
    Randomly flip RGB channels (forces model to learn semantic matching)

    Args:
        image: PIL Image (RGB)
        prob: Probability of flipping

    Returns:
        PIL Image (possibly with flipped RGB channels)
    """
    if random.random() > prob:
        return image

    # Random permutation of RGB channels
    r, g, b = image.split()
    channels = [r, g, b]
    random.shuffle(channels)
    return Image.merge('RGB', tuple(channels))


def random_horizontal_flip(image, prob=0.5):
    """
    Random horizontal flip

    Args:
        image: PIL Image
        prob: Probability of flipping

    Returns:
        PIL Image (possibly flipped)
    """
    if random.random() < prob:
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    return image


def apply_tps_transform(image, prob=0.3, strength=0.1):
    """
    Apply Thin Plate Spline (TPS) transformation (elastic deformation)

    This creates smooth, non-rigid deformations that prevent the model
    from memorizing exact spatial correspondences.

    Args:
        image: PIL Image
        prob: Probability of applying TPS
        strength: Deformation strength (0.0-1.0)
                  0.0 = no deformation
                  1.0 = strong deformation

    Returns:
        PIL Image (possibly deformed)
    """
    if random.random() > prob or strength == 0:
        return image

    try:
        import cv2
        from scipy.interpolate import Rbf
    except ImportError:
        # Fallback: skip TPS if scipy not available
        print("Warning: scipy not available, skipping TPS transform")
        return image

    # Convert to numpy
    img_array = np.array(image)
    h, w = img_array.shape[:2]

    # Create control points grid (3x3 grid for smooth deformation)
    num_points = 3

    # Source points (regular grid)
    src_points = []
    for i in range(num_points):
        for j in range(num_points):
            x = int(w * j / (num_points - 1))
            y = int(h * i / (num_points - 1))
            src_points.append([x, y])
    src_points = np.array(src_points, dtype=np.float32)

    # Destination points (perturbed grid)
    dst_points = src_points.copy()
    for i in range(len(dst_points)):
        # Skip corner points (keep them fixed for stability)
        if i in [0, num_points-1, len(dst_points)-num_points, len(dst_points)-1]:
            continue

        # Random perturbation
        max_shift = min(w, h) * strength * 0.15  # Max 15% shift at strength=1.0
        dx = random.uniform(-max_shift, max_shift)
        dy = random.uniform(-max_shift, max_shift)
        dst_points[i] += [dx, dy]

    # Clip to image bounds
    dst_points[:, 0] = np.clip(dst_points[:, 0], 0, w - 1)
    dst_points[:, 1] = np.clip(dst_points[:, 1], 0, h - 1)

    # Create TPS interpolator using RBF (Radial Basis Function)
    # This creates smooth deformations
    rbf_x = Rbf(dst_points[:, 0], dst_points[:, 1], src_points[:, 0], function='thin_plate', smooth=0)
    rbf_y = Rbf(dst_points[:, 0], dst_points[:, 1], src_points[:, 1], function='thin_plate', smooth=0)

    # Create dense mapping
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = rbf_x(grid_x, grid_y).astype(np.float32)
    map_y = rbf_y(grid_x, grid_y).astype(np.float32)

    # Apply mapping (remap)
    deformed = cv2.remap(img_array, map_x, map_y,
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)

    return Image.fromarray(deformed)


def augment_reference_image(image, config):
    """
    Apply all augmentations to reference image

    Args:
        image: PIL Image (RGB)
        config: Dictionary with augmentation parameters:
            {
                'color_jitter': bool,
                'hue_range': float (degrees),
                'saturation_range': tuple,
                'brightness_range': tuple,
                'rgb_flip_prob': float,
                'horizontal_flip_prob': float,
                'tps_prob': float,
                'tps_strength': float
            }

    Returns:
        Augmented PIL Image
    """
    if not config.get('enabled', True):
        return image

    # Color augmentation (apply in order)
    if config.get('color_jitter', True):
        # Hue shift
        if config.get('hue_range', 0) > 0:
            image = random_hue_shift(image, config['hue_range'])

        # Saturation
        saturation_range = config.get('saturation_range', (1.0, 1.0))
        if saturation_range != (1.0, 1.0):
            image = random_saturation(image, saturation_range)

        # Brightness
        brightness_range = config.get('brightness_range', (1.0, 1.0))
        if brightness_range != (1.0, 1.0):
            image = random_brightness(image, brightness_range)

    # RGB flip (strong augmentation)
    rgb_flip_prob = config.get('rgb_flip_prob', 0.0)
    if rgb_flip_prob > 0:
        image = random_rgb_flip(image, rgb_flip_prob)

    # Spatial augmentation
    # Horizontal flip
    horizontal_flip_prob = config.get('horizontal_flip_prob', 0.0)
    if horizontal_flip_prob > 0:
        image = random_horizontal_flip(image, horizontal_flip_prob)

    # TPS transform (most expensive, apply last)
    tps_prob = config.get('tps_prob', 0.0)
    tps_strength = config.get('tps_strength', 0.1)
    if tps_prob > 0 and tps_strength > 0:
        image = apply_tps_transform(image, tps_prob, tps_strength)

    return image


# Preset configurations for different training stages
AUGMENTATION_PRESETS = {
    'none': {
        'enabled': False
    },

    'minimal': {
        # Stage 1: Basic color augmentation only
        'enabled': True,
        'color_jitter': True,
        'hue_range': 20,
        'saturation_range': (0.8, 1.2),
        'brightness_range': (0.9, 1.1),
        'rgb_flip_prob': 0.0,
        'horizontal_flip_prob': 0.0,
        'tps_prob': 0.0,
        'tps_strength': 0.0
    },

    'moderate': {
        # Stage 2: Color + light spatial augmentation
        'enabled': True,
        'color_jitter': True,
        'hue_range': 30,
        'saturation_range': (0.7, 1.3),
        'brightness_range': (0.8, 1.2),
        'rgb_flip_prob': 0.1,
        'horizontal_flip_prob': 0.5,
        'tps_prob': 0.0,
        'tps_strength': 0.0
    },

    'strong': {
        # Stage 3: Full augmentation including TPS
        'enabled': True,
        'color_jitter': True,
        'hue_range': 30,
        'saturation_range': (0.7, 1.3),
        'brightness_range': (0.8, 1.2),
        'rgb_flip_prob': 0.15,
        'horizontal_flip_prob': 0.5,
        'tps_prob': 0.3,
        'tps_strength': 0.1
    }
}


def get_augmentation_config(preset='moderate', **kwargs):
    """
    Get augmentation configuration

    Args:
        preset: 'none', 'minimal', 'moderate', or 'strong'
        **kwargs: Override specific parameters

    Returns:
        Configuration dictionary
    """
    if preset not in AUGMENTATION_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(AUGMENTATION_PRESETS.keys())}")

    config = AUGMENTATION_PRESETS[preset].copy()
    config.update(kwargs)
    return config

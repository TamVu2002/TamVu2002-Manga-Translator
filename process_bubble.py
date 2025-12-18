"""
Bubble Processing Module - Optimized
Handles text removal and background color detection for manga speech bubbles.
"""
import cv2
import numpy as np


# ================== OPTIMIZED CONFIGURATION ==================
# Color detection
KMEANS_CLUSTERS = 3
KMEANS_ITERATIONS = 10
COLOR_QUANTIZATION_LEVELS = 32

# Threshold settings
LIGHT_BG_THRESHOLD = 180    # Above this = light background
DARK_BG_THRESHOLD = 80      # Below this = dark background
TEXT_FILTER_LIGHT = 180     # For light bubbles, keep pixels above this
TEXT_FILTER_DARK = 80       # For dark bubbles, keep pixels below this

# Bubble detection
DARK_BUBBLE_INTENSITY = 100  # Mean intensity below this = dark bubble

# Margin for center sampling
CENTER_MARGIN_RATIO = 0.2   # 20% margin from edges

# Morphology
MORPH_KERNEL_SIZE = 5       # Kernel size for morphological operations


def get_dominant_color(image, mask=None):
    """
    Get the dominant color using K-means clustering.
    """
    if mask is not None:
        pixels = image[mask == 255]
    else:
        pixels = image.reshape(-1, 3)
    
    if len(pixels) == 0:
        return (255, 255, 255)
    
    pixels = np.float32(pixels)
    k = min(KMEANS_CLUSTERS, len(pixels))
    
    if k < 1:
        return (255, 255, 255)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, KMEANS_ITERATIONS, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    unique, counts = np.unique(labels, return_counts=True)
    dominant_idx = unique[np.argmax(counts)]
    dominant_color = centers[dominant_idx]
    
    return tuple(int(c) for c in dominant_color)


def get_color_by_histogram(pixels, bins=32):
    """
    Find dominant color using histogram binning.
    """
    if len(pixels) == 0:
        return (255, 255, 255)
    
    hist_b = np.histogram(pixels[:, 0], bins=bins, range=(0, 256))[0]
    hist_g = np.histogram(pixels[:, 1], bins=bins, range=(0, 256))[0]
    hist_r = np.histogram(pixels[:, 2], bins=bins, range=(0, 256))[0]
    
    bin_width = 256 // bins
    b_peak = np.argmax(hist_b) * bin_width + bin_width // 2
    g_peak = np.argmax(hist_g) * bin_width + bin_width // 2
    r_peak = np.argmax(hist_r) * bin_width + bin_width // 2
    
    return (int(b_peak), int(g_peak), int(r_peak))


def get_color_by_mode(pixels):
    """
    Find the most frequent color using color quantization.
    """
    if len(pixels) == 0:
        return (255, 255, 255)
    
    # Quantize colors
    quantized = (pixels // 8) * 8
    
    # Use int32 to avoid overflow
    color_codes = quantized[:, 0].astype(np.int32) * 65536 + \
                  quantized[:, 1].astype(np.int32) * 256 + \
                  quantized[:, 2].astype(np.int32)
    
    unique, counts = np.unique(color_codes, return_counts=True)
    most_freq_code = unique[np.argmax(counts)]
    
    b = (most_freq_code // 65536) % 256
    g = (most_freq_code // 256) % 256
    r = most_freq_code % 256
    
    return (int(b), int(g), int(r))


def get_edge_color(image, edge_width=5):
    """
    Sample colors from the edges of the image.
    Useful when center might contain text.
    """
    h, w = image.shape[:2]
    
    # Sample from all 4 edges
    top = image[:edge_width, :].reshape(-1, 3)
    bottom = image[-edge_width:, :].reshape(-1, 3)
    left = image[:, :edge_width].reshape(-1, 3)
    right = image[:, -edge_width:].reshape(-1, 3)
    
    all_edge_pixels = np.vstack([top, bottom, left, right])
    
    if len(all_edge_pixels) == 0:
        return (255, 255, 255)
    
    return get_color_by_mode(all_edge_pixels)


def get_bubble_background_color(image, sample_border=True):
    """
    Detect speech bubble background color with improved accuracy.
    Uses multiple methods and voting.
    """
    h, w = image.shape[:2]
    
    # Calculate margins
    margin_y = max(5, int(h * CENTER_MARGIN_RATIO))
    margin_x = max(5, int(w * CENTER_MARGIN_RATIO))
    
    # Get center region (avoid edges and potential text)
    center_region = image[margin_y:h-margin_y, margin_x:w-margin_x]
    
    if center_region.size == 0:
        center_region = image
    
    center_pixels = center_region.reshape(-1, 3)
    
    # Determine if dark or light bubble
    gray_values = np.mean(center_pixels, axis=1)
    median_gray = np.median(gray_values)
    
    # Filter pixels based on bubble type
    if median_gray > 128:
        # Light bubble - keep bright pixels (background), remove dark (text)
        bg_mask = gray_values > TEXT_FILTER_LIGHT
    else:
        # Dark bubble - keep dark pixels (background), remove bright (text)
        bg_mask = gray_values < TEXT_FILTER_DARK
    
    # Apply mask
    if np.sum(bg_mask) > 50:
        bg_pixels = center_pixels[bg_mask]
    else:
        bg_pixels = center_pixels
    
    # Collect colors from multiple methods
    collected_colors = []
    
    # Method 1: Mode from filtered pixels (most reliable)
    collected_colors.append(get_color_by_mode(bg_pixels))
    
    # Method 2: Histogram peaks
    collected_colors.append(get_color_by_histogram(bg_pixels, bins=32))
    
    # Method 3: Median (good fallback)
    collected_colors.append(tuple(int(c) for c in np.median(bg_pixels, axis=0)))
    
    # Method 4: Edge sampling (if center is unreliable)
    if h > 20 and w > 20:
        collected_colors.append(get_edge_color(image))
    
    # Vote for best color (minimum distance to all others)
    collected_colors = np.array(collected_colors)
    
    best_color = collected_colors[0]
    min_total_dist = float('inf')
    
    for i, color in enumerate(collected_colors):
        total_dist = 0
        for j, other_color in enumerate(collected_colors):
            if i != j:
                dist = np.sum(np.abs(color.astype(np.int32) - other_color.astype(np.int32)))
                total_dist += dist
        
        if total_dist < min_total_dist:
            min_total_dist = total_dist
            best_color = color
    
    return tuple(int(c) for c in best_color)


def is_dark_bubble(image, threshold=None):
    """
    Determine if a bubble is dark (black bubble with white text).
    """
    if threshold is None:
        threshold = DARK_BUBBLE_INTENSITY
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    return mean_intensity < threshold


def create_bubble_mask(gray, fill_color, is_dark=False):
    """
    Create a mask for the bubble region using adaptive methods.
    """
    bg_intensity = np.mean(fill_color)
    
    if is_dark or bg_intensity < 50:
        # Dark bubble
        _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    elif bg_intensity > 200:
        # Light/white bubble
        _, thresh = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    else:
        # Medium color - use adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 5
        )
    
    # Clean up with morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return thresh


def process_dark_bubble(image, fill_color=None):
    """
    Process a dark speech bubble (black with white text).
    """
    if fill_color is None:
        fill_color = get_bubble_background_color(image)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create mask for dark region
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    
    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        h, w = image.shape[:2]
        largest_contour = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.int32)
        image[:] = fill_color
        return image, largest_contour, fill_color
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create filled mask
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, cv2.FILLED)
    
    # Dilate mask slightly to cover text edges
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    image[mask == 255] = fill_color
    
    return image, largest_contour, fill_color


def process_bubble(image, fill_color=None):
    """
    Process a light speech bubble, filling with background color.
    """
    if fill_color is None:
        fill_color = get_bubble_background_color(image)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create mask based on background intensity
    bg_intensity = np.mean(fill_color)
    
    if bg_intensity > 200: 
        # White/light background
        _, thresh = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    elif bg_intensity < 50:
        # Very dark background
        _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    else:
        # Medium color - adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 5
        )
    
    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        h, w = image.shape[:2]
        largest_contour = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.int32)
        image[:] = fill_color
        return image, largest_contour, fill_color
    
    largest_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, cv2.FILLED)
    
    # Dilate to cover text edges
    mask = cv2.dilate(mask, kernel, iterations=1)

    image[mask == 255] = fill_color

    return image, largest_contour, fill_color


def process_bubble_auto(image, force_dark=False, custom_color=None):
    """
    Automatically detect bubble type and process accordingly.
        
    Returns:
        tuple: (processed_image, contour, is_dark, detected_color)
    """
    # Auto-detect background color
    if custom_color is None:
        detected_color = get_bubble_background_color(image)
    else:
        detected_color = custom_color
    
    # Determine processing method
    if force_dark or is_dark_bubble(image):
        processed, contour, color_used = process_dark_bubble(image, detected_color)
        return processed, contour, True, color_used
    else:
        processed, contour, color_used = process_bubble(image, detected_color)
        return processed, contour, False, color_used


def process_bubble_inpaint(image, text_mask=None):
    """
    Process bubble using inpainting - preserves gradients and complex backgrounds.
    Best for stylized bubbles with special effects.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if text_mask is None:
        # Auto-detect text
        bg_color = get_bubble_background_color(image)
        bg_intensity = np.mean(bg_color)
        
        if bg_intensity > 128:
            # Light background, dark text
            _, text_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        else:
            # Dark background, light text
            _, text_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
    
    # Dilate mask to fully cover text
    kernel = np.ones((3, 3), np.uint8)
    text_mask_dilated = cv2.dilate(text_mask, kernel, iterations=2)
    
    # Inpaint
    result = cv2.inpaint(image, text_mask_dilated, 5, cv2.INPAINT_TELEA)
    
    # Find contour
    contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
    else:
        h, w = image.shape[:2]
        largest_contour = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.int32)
    
    return result, largest_contour


def process_bubble_preserve_gradient(image, text_mask=None):
    """
    Alias for process_bubble_inpaint for backwards compatibility.
    """
    return process_bubble_inpaint(image, text_mask)


def clean_bubble_edges(image, contour, edge_width=3):
    """
    Clean up edges of processed bubble to remove artifacts.
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    
    # Erode mask to get inner region
    kernel = np.ones((edge_width * 2 + 1, edge_width * 2 + 1), np.uint8)
    inner_mask = cv2.erode(mask, kernel, iterations=1)
    
    # Get edge region
    edge_mask = mask - inner_mask
    
    # Blur edge region
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Apply blurred edges
    result = image.copy()
    result[edge_mask == 255] = blurred[edge_mask == 255]
    
    return result

from ultralytics import YOLO
import numpy as np
import cv2

# Global cache for YOLO models to avoid reloading on every call
_yolo_model_cache = {}

# ================== OPTIMIZED CONFIGURATION ==================
# YOLO Detection Settings
YOLO_CONF_THRESHOLD = 0.15  # Lower = more bubbles detected (was implicit 0.25)
YOLO_IOU_THRESHOLD = 0.4    # NMS threshold
YOLO_MAX_DET = 300          # Max detections per image

# Long image handling
MAX_ASPECT_RATIO = 3.0      # When height/width > 3, start slicing
MIN_CHUNK_HEIGHT = 800      # Minimum chunk height in pixels
MAX_CHUNK_HEIGHT = 1500     # Target chunk height
GUTTER_MIN_HEIGHT = 10      # Minimum gutter height to consider valid
OVERLAP_SIZE = 200          # Fallback overlap if no gutter found
WHITE_THRESHOLD = 245       # Pixel value to consider "white"
BLACK_THRESHOLD = 15        # Pixel value to consider "black"
IOU_THRESHOLD = 0.45        # For removing duplicate detections (slightly lower)

# Black bubble detection - OPTIMIZED
BLACK_BUBBLE_THRESHOLD = 70     # Max intensity for black regions (was 50)
BLACK_BUBBLE_MIN_AREA = 800     # Minimum area in pixels (was 1000)
BLACK_BUBBLE_MAX_AREA_RATIO = 0.5  # Maximum bubble area relative to image (was 0.4)
BLACK_BUBBLE_MIN_ASPECT = 0.15  # Minimum width/height ratio (was 0.2)
BLACK_BUBBLE_MAX_ASPECT = 6.0   # Maximum width/height ratio (was 5.0)
BLACK_BUBBLE_FILL_RATIO = 0.25  # Minimum fill ratio (was 0.3)

# Additional bubble types
GRAY_BUBBLE_MIN = 30        # For gray bubbles
GRAY_BUBBLE_MAX = 100       # For gray bubbles

# Edge detection for borderless bubbles
EDGE_BUBBLE_MIN_AREA = 1500
EDGE_BUBBLE_CANNY_LOW = 50
EDGE_BUBBLE_CANNY_HIGH = 150


def detect_black_bubbles(image, min_area=None, max_area_ratio=None):
    """
    Detect black/dark speech bubbles using OpenCV contour detection.
    Optimized for better detection rate.
    """
    if min_area is None:
        min_area = BLACK_BUBBLE_MIN_AREA
    if max_area_ratio is None:
        max_area_ratio = BLACK_BUBBLE_MAX_AREA_RATIO
    
    height, width = image.shape[:2]
    max_area = int(width * height * max_area_ratio)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    detections = []
    
    # Method 1: Dark region detection (original, optimized thresholds)
    _, thresh_dark = cv2.threshold(gray, BLACK_BUBBLE_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological operations - slightly larger kernel for better merging
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thresh_dark = cv2.morphologyEx(thresh_dark, cv2.MORPH_CLOSE, kernel)
    thresh_dark = cv2.morphologyEx(thresh_dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    
    contours_dark, _ = cv2.findContours(thresh_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours_dark:
        det = _validate_bubble_contour(contour, gray, min_area, max_area, is_dark=True)
        if det:
            detections.append(det)
    
    # Method 2: Gray bubble detection (medium dark)
    _, thresh_gray = cv2.threshold(gray, GRAY_BUBBLE_MAX, 255, cv2.THRESH_BINARY_INV)
    thresh_gray_light = cv2.threshold(gray, GRAY_BUBBLE_MIN, 255, cv2.THRESH_BINARY)[1]
    thresh_gray = cv2.bitwise_and(thresh_gray, thresh_gray_light)
    
    kernel_gray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh_gray = cv2.morphologyEx(thresh_gray, cv2.MORPH_CLOSE, kernel_gray)
    
    contours_gray, _ = cv2.findContours(thresh_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours_gray:
        det = _validate_bubble_contour(contour, gray, min_area * 1.5, max_area, is_dark=True)
        if det:
            # Avoid duplicates with dark bubbles
            if not _overlaps_existing(det, detections):
                detections.append(det)
    
    # Method 3: Adaptive threshold for gradient bubbles
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 21, 10)
    
    # Focus on darker areas
    _, dark_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    adaptive_thresh = cv2.bitwise_and(adaptive_thresh, dark_mask)
    
    kernel_adapt = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    adaptive_thresh = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel_adapt)
    
    contours_adaptive, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours_adaptive:
        det = _validate_bubble_contour(contour, gray, min_area * 2, max_area, is_dark=True)
        if det and not _overlaps_existing(det, detections):
            detections.append(det)
    
    return detections


def detect_borderless_bubbles(image):
    """
    Detect bubbles without clear borders using edge detection.
    Useful for stylized manga with open/borderless speech areas.
    """
    height, width = image.shape[:2]
    max_area = int(width * height * 0.3)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect edges
    edges = cv2.Canny(gray, EDGE_BUBBLE_CANNY_LOW, EDGE_BUBBLE_CANNY_HIGH)
    
    # Dilate edges to close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours from edges
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area < EDGE_BUBBLE_MIN_AREA or area > max_area:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        
        # Check aspect ratio
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < 0.2 or aspect_ratio > 5.0:
            continue
        
        # Check if interior is relatively uniform (text area)
        roi = gray[y:y+h, x:x+w]
        std_dev = np.std(roi)
        
        # Low std_dev means uniform (likely text area after text)
        # High std_dev with specific pattern might be text
        mean_val = np.mean(roi)
        
        # Look for white/light areas (typical bubble interior)
        if mean_val > 180 and std_dev < 60:
            confidence = min(0.6, 0.4 + (mean_val - 180) / 150)
            detections.append([x, y, x + w, y + h, confidence, 0])
    
    return detections


def _validate_bubble_contour(contour, gray, min_area, max_area, is_dark=False):
    """
    Validate a contour and return detection if it looks like a bubble.
    Returns [x1, y1, x2, y2, confidence, class_id] or None
    """
    area = cv2.contourArea(contour)
    
    if area < min_area or area > max_area:
        return None
    
    x, y, w, h = cv2.boundingRect(contour)
    
    # Filter by aspect ratio
    aspect_ratio = w / h if h > 0 else 0
    if aspect_ratio < BLACK_BUBBLE_MIN_ASPECT or aspect_ratio > BLACK_BUBBLE_MAX_ASPECT:
        return None
    
    # Check fill ratio
    rect_area = w * h
    fill_ratio = area / rect_area if rect_area > 0 else 0
    if fill_ratio < BLACK_BUBBLE_FILL_RATIO:
        return None
    
    # Verify intensity
    roi = gray[y:y+h, x:x+w]
    mean_intensity = np.mean(roi)
    
    if is_dark:
        # For dark bubbles, verify it's actually dark
        if mean_intensity > BLACK_BUBBLE_THRESHOLD + 40:
            return None
        # Higher confidence for darker bubbles
        darkness_score = 1 - (mean_intensity / (BLACK_BUBBLE_THRESHOLD + 40))
    else:
        darkness_score = 0.5
    
    # Calculate confidence
    confidence = min(0.85, fill_ratio * 0.5 + darkness_score * 0.5)
    
    return [x, y, x + w, y + h, confidence, 0]


def _overlaps_existing(new_det, existing_dets, threshold=0.5):
    """Check if new detection overlaps significantly with existing ones."""
    for det in existing_dets:
        iou = calculate_iou(new_det, det)
        if iou > threshold:
            return True
    return False


def find_safe_cut_points(image, target_height=MAX_CHUNK_HEIGHT):
    """
    Find safe places to cut the image (white/black gutters between panels).
    """
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate mean intensity for each row
    row_means = np.mean(gray, axis=1)
    
    # Find rows that are mostly white or mostly black (gutters)
    is_gutter = (row_means > WHITE_THRESHOLD) | (row_means < BLACK_THRESHOLD)
    
    # Find continuous gutter regions
    gutter_regions = []
    start = None
    
    for i, is_gut in enumerate(is_gutter):
        if is_gut and start is None:
            start = i
        elif not is_gut and start is not None:
            if i - start >= GUTTER_MIN_HEIGHT:
                gutter_regions.append((start, i, (start + i) // 2))
            start = None
    
    if start is not None and height - start >= GUTTER_MIN_HEIGHT:
        gutter_regions.append((start, height, (start + height) // 2))
    
    if not gutter_regions:
        return []
    
    cut_points = []
    last_cut = 0
    
    for start, end, center in gutter_regions:
        if center - last_cut >= MIN_CHUNK_HEIGHT:
            if center - last_cut >= target_height * 0.7:
                cut_points.append(center)
                last_cut = center
    
    return cut_points


def calculate_iou(box1, box2):
    """Calculate Intersection over Union of two boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def remove_duplicate_detections(detections, iou_threshold=IOU_THRESHOLD):
    """Remove duplicate detections based on IoU, keeping higher confidence ones."""
    if len(detections) <= 1:
        return detections
    
    # Sort by confidence descending
    sorted_dets = sorted(detections, key=lambda x: x[4], reverse=True)
    
    keep = []
    while sorted_dets:
        best = sorted_dets.pop(0)
        keep.append(best)
        
        sorted_dets = [
            det for det in sorted_dets 
            if calculate_iou(best, det) < iou_threshold
        ]
    
    return keep


def detect_bubbles_on_chunks(model, image, cut_points):
    """Detect bubbles on image chunks and merge results."""
    height = image.shape[0]
    all_detections = []
    boundaries = [0] + cut_points + [height]
    
    print(f"Processing image in {len(boundaries) - 1} chunks...")
    
    for i in range(len(boundaries) - 1):
        y_start = boundaries[i]
        y_end = boundaries[i + 1]
        chunk = image[y_start:y_end]
        
        if chunk.shape[0] < 50:
            continue
        
        # YOLO detection with optimized settings
        results = model(chunk, verbose=False, conf=YOLO_CONF_THRESHOLD, 
                       iou=YOLO_IOU_THRESHOLD, max_det=YOLO_MAX_DET)[0]
        chunk_detections = results.boxes.data.tolist()
        
        # Adjust y-coordinates
        for det in chunk_detections:
            det[1] += y_start
            det[3] += y_start
            all_detections.append(det)
        
        print(f"  Chunk {i+1}: y={y_start}-{y_end}, found {len(chunk_detections)} bubbles")
    
    merged = remove_duplicate_detections(all_detections)
    print(f"Total: {len(all_detections)} detections → {len(merged)} after merge")
    
    return merged


def detect_bubbles_with_fallback(model, image):
    """Detect bubbles using overlap-based slicing when no gutters found."""
    height = image.shape[0]
    all_detections = []
    
    chunk_height = MAX_CHUNK_HEIGHT
    overlap = OVERLAP_SIZE
    
    y = 0
    chunk_num = 0
    
    print(f"No gutters found. Using overlap-based slicing...")
    
    while y < height:
        y_end = min(y + chunk_height, height)
        chunk = image[y:y_end]
        
        if chunk.shape[0] < 50:
            break
        
        results = model(chunk, verbose=False, conf=YOLO_CONF_THRESHOLD,
                       iou=YOLO_IOU_THRESHOLD, max_det=YOLO_MAX_DET)[0]
        chunk_detections = results.boxes.data.tolist()
        
        for det in chunk_detections:
            det[1] += y
            det[3] += y
            all_detections.append(det)
        
        chunk_num += 1
        print(f"  Chunk {chunk_num}: y={y}-{y_end}, found {len(chunk_detections)} bubbles")
        
        y = y_end - overlap
        if y_end >= height:
            break
    
    merged = remove_duplicate_detections(all_detections)
    print(f"Total: {len(all_detections)} detections → {len(merged)} after merge")
    
    return merged


def detect_bubbles_multiscale(model, image):
    """
    Multi-scale detection for better coverage of various bubble sizes.
    """
    all_detections = []
    
    scales = [1.0, 0.75, 1.25]  # Original, smaller, larger
    
    for scale in scales:
        if scale == 1.0:
            scaled_image = image
        else:
            new_width = int(image.shape[1] * scale)
            new_height = int(image.shape[0] * scale)
            scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        results = model(scaled_image, verbose=False, conf=YOLO_CONF_THRESHOLD,
                       iou=YOLO_IOU_THRESHOLD, max_det=YOLO_MAX_DET)[0]
        
        for det in results.boxes.data.tolist():
            # Scale coordinates back to original size
            if scale != 1.0:
                det[0] /= scale  # x1
                det[1] /= scale  # y1
                det[2] /= scale  # x2
                det[3] /= scale  # y2
            all_detections.append(det)
    
    # Remove duplicates from multi-scale
    merged = remove_duplicate_detections(all_detections, iou_threshold=0.5)
    return merged


def detect_bubbles(model_path, image_input, enable_black_bubble=True, use_multiscale=False):
    """
    Detects bubbles in an image using YOLOv8 model + OpenCV fallback.
    
    Args:
        model_path (str): Path to the YOLO model
        image_input: File path OR numpy array (BGR)
        enable_black_bubble (bool): Enable OpenCV black bubble detection
        use_multiscale (bool): Use multi-scale detection (slower but more accurate)

    Returns:
        list: Detections with [x1, y1, x2, y2, score, class_id, is_dark_bubble]
    """
    global _yolo_model_cache
    
    # Cache model
    if model_path not in _yolo_model_cache:
        print(f"Loading YOLO model from {model_path}...")
        _yolo_model_cache[model_path] = YOLO(model_path)
        print("YOLO model loaded and cached!")
    
    model = _yolo_model_cache[model_path]
    
    # Load image if path
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
    else:
        image = image_input
    
    if image is None:
        return []
    
    height, width = image.shape[:2]
    aspect_ratio = height / width
    
    # Get YOLO detections
    if aspect_ratio > MAX_ASPECT_RATIO:
        print(f"Long image detected: {width}x{height} (ratio: {aspect_ratio:.1f})")
        cut_points = find_safe_cut_points(image)
        
        if cut_points:
            print(f"Found {len(cut_points)} safe cut points")
            yolo_detections = detect_bubbles_on_chunks(model, image, cut_points)
        else:
            yolo_detections = detect_bubbles_with_fallback(model, image)
    elif use_multiscale:
        yolo_detections = detect_bubbles_multiscale(model, image)
    else:
        # Normal detection with optimized parameters
        results = model(image, verbose=False, conf=YOLO_CONF_THRESHOLD,
                       iou=YOLO_IOU_THRESHOLD, max_det=YOLO_MAX_DET)[0]
        yolo_detections = results.boxes.data.tolist()
    
    # Get additional detections
    additional_detections = []
    
    if enable_black_bubble:
        black_detections = detect_black_bubbles(image)
        if black_detections:
            print(f"OpenCV found {len(black_detections)} potential dark bubbles")
            for det in black_detections:
                det.append(1)  # is_dark_bubble = 1
            additional_detections.extend(black_detections)
        
    # Mark YOLO detections
    for det in yolo_detections:
        if len(det) == 6:
            det.append(0)  # is_dark_bubble = 0
        
    # Merge all detections
    if additional_detections:
        all_detections = yolo_detections + additional_detections
        merged = remove_duplicate_detections(all_detections)
        print(f"Total: {len(yolo_detections)} YOLO + {len(additional_detections)} OpenCV = {len(merged)} after merge")
        return merged
    
    return yolo_detections


def detect_bubbles_aggressive(model_path, image_input):
    """
    Aggressive detection mode - finds maximum bubbles at cost of speed.
    Use when normal detection misses too many bubbles.
    """
    return detect_bubbles(model_path, image_input, enable_black_bubble=True, use_multiscale=True)


def clear_model_cache():
    """Clear the YOLO model cache to free memory."""
    global _yolo_model_cache
    _yolo_model_cache.clear()

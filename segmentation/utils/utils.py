import random
import json
from dataclasses import dataclass, asdict
from typing import Any, List, Dict, Optional, Union, Tuple
from pathlib import Path

import cv2
import torch
import requests
import numpy as np
from PIL import Image
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline


# --- Result utils ---
@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))
    
# --- Plot utils ---
def annotate(image: Union[Image.Image, np.ndarray], detection_results: List[DetectionResult]) -> np.ndarray:
    # Convert PIL Image to OpenCV format
    image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    # Iterate over detections and add bounding boxes and masks
    for detection in detection_results:
        label = detection.label
        score = detection.score
        box = detection.box
        mask = detection.mask

        # Sample a random color for each detection
        color = np.random.randint(0, 256, size=3)

        # Draw bounding box
        cv2.rectangle(image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), color.tolist(), 2)
        cv2.putText(image_cv2, f'{label}: {score:.2f}', (box.xmin, box.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

        # If mask is available, apply it
        if mask is not None:
            # Convert mask to uint8
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

    return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

def plot_detections(
    image: Union[Image.Image, np.ndarray],
    detections: List[DetectionResult],
    save_name: Optional[str] = None
) -> None:
    annotated_image = annotate(image, detections)
    plt.imshow(annotated_image)
    plt.axis('off')
    if save_name:
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()


def random_named_css_colors(num_colors: int) -> List[str]:
    """
    Returns a list of randomly selected named CSS colors.

    Args:
    - num_colors (int): Number of random colors to generate.

    Returns:
    - list: List of randomly selected named CSS colors.
    """
    # List of named CSS colors
    named_css_colors = [
        'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond',
        'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
        'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey',
        'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
        'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
        'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite',
        'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory',
        'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow',
        'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray',
        'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine',
        'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise',
        'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive',
        'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip',
        'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown',
        'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey',
        'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white',
        'whitesmoke', 'yellow', 'yellowgreen'
    ]

    # Sample random named CSS colors
    return random.sample(named_css_colors, min(num_colors, len(named_css_colors)))


# --- Utils ---
def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon

def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask

def load_image(image_str: str) -> Image.Image:
    if image_str.startswith("http"):
        image = Image.open(requests.get(image_str, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_str).convert("RGB")

    return image

def get_boxes(results: DetectionResult) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)

    return [boxes]

def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks


# --- Save ---
def save_detection_results(detections: List[DetectionResult], output_path: str) -> None:
    """Save detection results (bounding boxes and labels) to JSON file."""
    results_data = []
    for detection in detections:
        result_dict = {
            'label': detection.label,
            'score': detection.score,
            'box': {
                'xmin': detection.box.xmin,
                'ymin': detection.box.ymin,
                'xmax': detection.box.xmax,
                'ymax': detection.box.ymax
            }
        }
        results_data.append(result_dict)

    with open(output_path, 'w') as f:
        json.dump(results_data, f, indent=2)


def save_segmentation_masks(detections: List[DetectionResult], output_dir: str, base_name: str) -> List[str]:
    """Save individual segmentation masks and return list of saved file paths."""
    saved_paths = []
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    for i, detection in enumerate(detections):
        if detection.mask is not None:
            mask_filename = f"{base_name}_mask_{i}_{detection.label.replace(' ', '_')}.png"
            mask_path = output_path / mask_filename

            # Ensure mask is in proper format for saving
            # Convert to binary mask (0 or 255)
            binary_mask = (detection.mask > 0).astype(np.uint8) * 255

            # Convert to PIL Image and save
            mask_img = Image.fromarray(binary_mask, mode='L')  # 'L' mode for grayscale
            mask_img.save(mask_path)
            saved_paths.append(str(mask_path))

    return saved_paths


def save_combined_mask(detections: List[DetectionResult], output_path: str) -> None:
    """Save a combined mask with all detections as black and white."""
    if not detections or all(d.mask is None for d in detections):
        return

    # Get image dimensions from first mask
    first_mask = next(d.mask for d in detections if d.mask is not None)
    combined_mask = np.zeros_like(first_mask, dtype=np.uint8)

    # Combine all masks as binary (white where any mask exists)
    for detection in detections:
        if detection.mask is not None:
            binary_detection = (detection.mask > 0).astype(np.uint8)
            combined_mask = np.where(binary_detection > 0, 255, combined_mask)
    
    # Save combined mask
    combined_img = Image.fromarray(combined_mask, mode='L')
    combined_img.save(output_path)


def save_background_image(image: np.ndarray, detections: List[DetectionResult], output_path: str) -> None:
    """Save background image with segmented parts made black"""
    if not detections or all(d.mask is None for d in detections):
        # No masks, save original image
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(output_path)
        return

    # Use the same logic as save_combined_mask
    first_mask = next(d.mask for d in detections if d.mask is not None)
    combined_mask = np.zeros_like(first_mask, dtype=np.uint8)

    # Combine all masks as binary (white where any mask exists)
    for detection in detections:
        if detection.mask is not None:
            binary_detection = (detection.mask > 0).astype(np.uint8)
            combined_mask = np.where(binary_detection > 0, 255, combined_mask)

    # Create background image: original with segmented parts made black
    background_image = image.copy()
    mask_boolean = combined_mask > 0
    background_image[mask_boolean] = [0, 0, 0]

    # Save background image
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(background_image).save(output_path)


def save_objects_only(image: np.ndarray, detections: List[DetectionResult], output_path: str) -> None:
    """Save only segmented objects with transparent background using the same logic as save_combined_mask"""
    if not detections or all(d.mask is None for d in detections):
        # No masks, save empty transparent image
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        transparent_img = Image.new('RGBA', (image.shape[1], image.shape[0]), (0, 0, 0, 0))
        transparent_img.save(output_path)
        return

    # Use the same logic as save_combined_mask
    first_mask = next(d.mask for d in detections if d.mask is not None)
    combined_mask = np.zeros_like(first_mask, dtype=np.uint8)

    # Combine all masks as binary (white where any mask exists)
    for detection in detections:
        if detection.mask is not None:
            binary_detection = (detection.mask > 0).astype(np.uint8)
            combined_mask = np.where(binary_detection > 0, 255, combined_mask)

    # Create RGBA image with transparency
    rgba_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)

    # Copy RGB channels from original image
    rgba_image[:, :, :3] = image

    # Set alpha channel: 255 (opaque) where mask exists, 0 (transparent) elsewhere
    mask_boolean = combined_mask > 0
    rgba_image[:, :, 3] = np.where(mask_boolean, 255, 0)

    # Save as PNG with transparency
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba_image, mode='RGBA').save(output_path)


def save_detection_visualization(image: np.ndarray, detections: List[DetectionResult],
                                output_path: str, show_boxes_only: bool = False) -> None:
    """Save detection visualization with bounding boxes and/or masks."""
    if show_boxes_only:
        # Create a copy for boxes-only visualization
        detections_boxes_only = []
        for detection in detections:
            det_copy = DetectionResult(
                score=detection.score,
                label=detection.label,
                box=detection.box,
                mask=None  # Remove mask for boxes-only visualization
            )
            detections_boxes_only.append(det_copy)
        annotated_image = annotate(image, detections_boxes_only)
    else:
        annotated_image = annotate(image, detections)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis('off')
    plt.title('Detections' if not show_boxes_only else 'Bounding Boxes Only')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()


def save_all_intermediate_results(image: Union[Image.Image, np.ndarray],
                                 detections: List[DetectionResult],
                                 output_dir: str, base_name: str) -> Dict[str, Any]:
    """Save detection results (JSON), combined mask, and visualization with masks."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    image_array = np.array(image) if isinstance(image, Image.Image) else image

    saved_files = {
        'Detection results (json)': None,
        'Detection results (img)': None,
        'Mask': None,
        'Background': None,
        'Objects': None,
    }

    # Create subdirectories
    (output_path / "detection_results" / "json").mkdir(parents=True, exist_ok=True)
    (output_path / "detection_results" / "imgs").mkdir(parents=True, exist_ok=True)
    (output_path / "mask").mkdir(parents=True, exist_ok=True)
    (output_path / "background").mkdir(parents=True, exist_ok=True)
    (output_path / "object").mkdir(parents=True, exist_ok=True)

    # Save detection results (JSON)
    detection_json_path = output_path / "detection_results" / "json" / f"{base_name}.json"
    save_detection_results(detections, str(detection_json_path))
    saved_files['Detection results (json)'] = str(detection_json_path)

    # Save combined mask
    combined_mask_path = output_path / "mask" / f"{base_name}.png"
    save_combined_mask(detections, str(combined_mask_path))
    saved_files['Mask'] = str(combined_mask_path)

    # Save visualization with masks
    viz_with_masks_path = output_path / "detection_results" / "imgs" / f"{base_name}.png"
    save_detection_visualization(image_array, detections, str(viz_with_masks_path), show_boxes_only=False)
    saved_files['Detection results (img)'] = str(viz_with_masks_path)

    # Save background image
    background_path = output_path / "background" / f"{base_name}.png"
    save_background_image(image_array, detections, str(background_path))
    saved_files['Background'] = str(background_path)

    # Save objects only (transparent background)
    objects_only_path = output_path / "object" / f"{base_name}.png"
    save_objects_only(image_array, detections, str(objects_only_path))
    saved_files['Objects'] = str(objects_only_path)

    return saved_files
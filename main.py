"""
1. Grounding DINO) Detect a given set of texts in the img. The output is a set of bboxes.
2. SAM) Prompt SAM w the bboxes, for which the model will output segmentation masks.
"""
from typing import Any, List, Dict, Optional, Union, Tuple
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline

from utils import *


# --- Config ---
image_path = "/Users/jiyoonjeon/projects/DynamicAlbum/data/album/4 ONLY.jpg"  
output_dir = "/Users/jiyoonjeon/projects/DynamicAlbum/data"
labels = ["bubble"]

threshold = 0.3
detector_id = "IDEA-Research/grounding-dino-tiny"
segmenter_id = "facebook/sam-vit-base"


# --- Utils ---
def detect(
    image: Image.Image,
    labels: List[str],
    threshold: float = 0.3,
    detector_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
    object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)

    labels = [label if label.endswith(".") else label + "." for label in labels]

    results = object_detector(image, candidate_labels=labels, threshold=threshold)
    results = [DetectionResult.from_dict(result) for result in results]

    return results


def segment(
    image: Image.Image,
    detection_results: List[Dict[str, Any]],
    polygon_refinement: bool = False,
    segmenter_id: Optional[str] = None
) -> List[DetectionResult]:
    """
    Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-base"

    segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
    processor = AutoProcessor.from_pretrained(segmenter_id)

    boxes = get_boxes(detection_results)
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]

    masks = refine_masks(masks, polygon_refinement)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    return detection_results


def grounded_segmentation(
    image: Union[Image.Image, str],
    labels: List[str],
    threshold: float = 0.3,
    polygon_refinement: bool = False,
    detector_id: Optional[str] = None,
    segmenter_id: Optional[str] = None
) -> Tuple[np.ndarray, List[DetectionResult]]:
    if isinstance(image, str):
        image = load_image(image)

    detections = detect(image, labels, threshold, detector_id)
    detections = segment(image, detections, polygon_refinement, segmenter_id)

    return np.array(image), detections


# --- Inference ---
def main():
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"Processing image: {image_path}")
    print(f"Labels to detect: {labels}")
    print(f"Threshold: {threshold}")
    print(f"Output directory: {output_dir}")

    # Run grounded segmentation
    image_array, detections = grounded_segmentation(
        image=image_path,
        labels=labels,
        threshold=threshold,
        polygon_refinement=True,
        detector_id=detector_id,
        segmenter_id=segmenter_id
    )

    print(f"✅ Found {len(detections)} detections")

    # Generate base name for all output files
    input_name = Path(image_path).stem

    # Save all intermediate results
    print("🎧 Saving all intermediate results...")
    saved_files = save_all_intermediate_results(
        image=image_array,
        detections=detections,
        output_dir=output_dir,
        base_name=input_name
    )

    # Print summary of saved files
    print("\n=== Saved Files Summary ===")
    for file_type, file_path in saved_files.items():
        print(f"{file_type}: {file_path}")
    print("\n===========================")

if __name__ == "__main__":
    main()
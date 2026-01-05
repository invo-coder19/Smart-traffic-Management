"""
Utility functions for image processing and helper operations
"""

import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import config


class ImageProcessor:
    """Helper class for image processing operations"""
    
    @staticmethod
    def resize_image(image, max_size=config.MAX_IMAGE_SIZE):
        """
        Resize image while maintaining aspect ratio
        
        Args:
            image: Input image (numpy array or PIL Image)
            max_size: Tuple of (max_width, max_height)
        
        Returns:
            Resized image
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        h, w = image.shape[:2]
        max_w, max_h = max_size
        
        # Calculate scaling factor
        scale = min(max_w / w, max_h / h, 1.0)
        
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return image
    
    @staticmethod
    def enhance_image(image):
        """
        Enhance image for better detection
        
        Args:
            image: Input image (numpy array)
        
        Returns:
            Enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels
        enhanced = cv2.merge([l, a, b])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    @staticmethod
    def draw_bounding_box(image, bbox, label, color=(0, 255, 0), thickness=2):
        """
        Draw bounding box with label on image
        
        Args:
            image: Input image
            bbox: Bounding box as (x, y, w, h)
            label: Text label
            color: Box color in BGR
            thickness: Line thickness
        
        Returns:
            Image with bounding box
        """
        x, y, w, h = bbox
        
        # Draw rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
        
        # Draw label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x, y - label_size[1] - 10), 
                     (x + label_size[0], y), color, -1)
        
        # Draw label text
        cv2.putText(image, label, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image
    
    @staticmethod
    def annotate_violation(image, violation_type, confidence, bbox=None):
        """
        Annotate image with violation information
        
        Args:
            image: Input image
            violation_type: Type of violation detected
            confidence: Detection confidence score
            bbox: Optional bounding box
        
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        # Get violation info
        violation_info = config.VIOLATION_TYPES.get(violation_type, {})
        violation_name = violation_info.get("name", violation_type)
        
        # Determine color based on severity
        severity = violation_info.get("severity", "NONE")
        if severity == "HIGH":
            color = (0, 0, 255)  # Red
        elif severity == "MEDIUM":
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 255, 0)  # Green
        
        # Draw bounding box if provided
        if bbox:
            annotated = ImageProcessor.draw_bounding_box(
                annotated, bbox, f"{violation_name} ({confidence:.0%})", color
            )
        
        # Add violation text at top
        text = f"{violation_name} - Confidence: {confidence:.0%}"
        cv2.putText(annotated, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return annotated
    
    @staticmethod
    def convert_to_grayscale(image):
        """Convert image to grayscale"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    @staticmethod
    def get_timestamp():
        """Get current timestamp"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    @staticmethod
    def pil_to_cv2(pil_image):
        """Convert PIL Image to OpenCV format"""
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    @staticmethod
    def cv2_to_pil(cv2_image):
        """Convert OpenCV image to PIL format"""
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

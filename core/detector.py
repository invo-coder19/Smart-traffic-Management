"""
Traffic Violation Detector Module
Implements rule-based detection for common Indian traffic violations
"""

import cv2
import numpy as np
from PIL import Image
import config
from core.utils import ImageProcessor


class ViolationDetector:
    """Detects traffic violations using computer vision"""
    
    def __init__(self):
        """Initialize detector with configuration"""
        self.image_processor = ImageProcessor()
        self.detection_methods = {
            'HELMETLESS': self._detect_helmetless,
            'TRIPLE_RIDING': self._detect_triple_riding,
            'SIGNAL_JUMP': self._detect_signal_jump,
            'OVER_SPEEDING': self._detect_over_speeding
        }
    
    def _detect_helmetless(self, image):
        """
        Detect helmetless riding using head region and color analysis
        
        Args:
            image: Input image (numpy array)
        
        Returns:
            (violation_detected, confidence, bbox)
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Skin color range (for detecting visible head/face)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask for skin color
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours in upper half of image (where heads typically are)
        h, w = image.shape[:2]
        upper_mask = np.zeros_like(skin_mask)
        upper_mask[:int(h*0.6), :] = skin_mask[:int(h*0.6), :]
        
        contours, _ = cv2.findContours(upper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, 0.0, None
        
        # Find largest skin-colored region
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # If significant skin region detected in upper portion, likely no helmet
        total_area = h * w
        skin_ratio = area / total_area
        
        if skin_ratio > config.HELMET_COLOR_THRESHOLD:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Calculate confidence based on skin area ratio
            confidence = min(skin_ratio * 2.5, 0.95)
            
            return True, confidence, (x, y, w, h)
        
        return False, skin_ratio * 2, None
    
    def _detect_triple_riding(self, image):
        """
        Detect triple riding by counting persons
        Uses contour analysis as a simple approach
        
        Args:
            image: Input image
        
        Returns:
            (violation_detected, confidence, bbox)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area (potential persons)
        h, w = image.shape[:2]
        min_area = (h * w) * 0.02  # At least 2% of image
        max_area = (h * w) * 0.4   # At most 40% of image
        
        person_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # Check aspect ratio (persons are typically vertical)
                x, y, w_c, h_c = cv2.boundingRect(contour)
                aspect_ratio = h_c / float(w_c) if w_c > 0 else 0
                
                if 0.5 < aspect_ratio < 4:  # Reasonable person aspect ratio
                    person_contours.append(contour)
        
        person_count = len(person_contours)
        
        if person_count > config.PERSON_COUNT_THRESHOLD:
            # Get bounding box for all persons
            all_points = np.vstack([cv2.convexHull(c) for c in person_contours])
            x, y, w, h = cv2.boundingRect(all_points)
            
            # Confidence increases with person count
            confidence = min(0.6 + (person_count - 2) * 0.15, 0.95)
            
            return True, confidence, (x, y, w, h)
        
        # Return low confidence if 2 or fewer detected
        return False, 0.3, None
    
    def _detect_signal_jump(self, image):
        """
        Detect signal jumping (simulated - would need video/motion analysis in real implementation)
        This is a placeholder using red color detection for traffic light
        
        Args:
            image: Input image
        
        Returns:
            (violation_detected, confidence, bbox)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Red color ranges (traffic light red)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Find red regions in upper portion (where signals are)
        h, w = image.shape[:2]
        upper_red_mask = np.zeros_like(red_mask)
        upper_red_mask[:int(h*0.4), :] = red_mask[:int(h*0.4), :]
        
        # Find contours
        contours, _ = cv2.findContours(upper_red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find circular red objects (traffic lights)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Minimum size for signal
                    # Check circularity
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                        
                        if circularity > 0.6:  # Reasonably circular
                            x, y, w, h = cv2.boundingRect(contour)
                            
                            # Simulated detection (in real scenario, check vehicle motion)
                            confidence = 0.7
                            return True, confidence, (x, y, w, h)
        
        return False, 0.2, None
    
    def _detect_over_speeding(self, image):
        """
        Detect over-speeding (simulated)
        Real implementation would need:
        - Multiple frames for motion analysis
        - Distance calculation between frames
        - Time delta between frames
        
        Args:
            image: Input image
        
        Returns:
            (violation_detected, confidence, bbox)
        """
        # This is a simulation - randomly detect with low confidence
        # In real scenario, this would analyze motion vectors or use speed sensors
        
        # Apply motion blur detection as a proxy
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate image variance (lower = more blur = potentially faster motion)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # If image has significant blur (low variance), might indicate speed
        if laplacian_var < 100:  # Threshold for blur
            h, w = image.shape[:2]
            # Return whole image as bbox
            confidence = 0.6
            return True, confidence, (0, 0, w, h)
        
        return False, 0.3, None
    
    def detect_violations(self, image):
        """
        Detect all violations in the image
        
        Args:
            image: Input image (numpy array or PIL Image)
        
        Returns:
            Dictionary with violation details
        """
        # Convert PIL to OpenCV if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Enhance image for better detection
        enhanced = self.image_processor.enhance_image(image)
        
        # Initialize results
        results = {
            'violations_detected': [],
            'primary_violation': None,
            'confidence': 0.0,
            'bbox': None,
            'annotated_image': image.copy()
        }
        
        # Check each violation type
        for violation_type, detect_func in self.detection_methods.items():
            detected, confidence, bbox = detect_func(enhanced)
            
            if detected and confidence > config.DETECTION_CONFIDENCE_THRESHOLD:
                results['violations_detected'].append({
                    'type': violation_type,
                    'confidence': confidence,
                    'bbox': bbox
                })
        
        # If violations detected, select primary one (highest confidence)
        if results['violations_detected']:
            primary = max(results['violations_detected'], key=lambda x: x['confidence'])
            results['primary_violation'] = primary['type']
            results['confidence'] = primary['confidence']
            results['bbox'] = primary['bbox']
            
            # Annotate image
            results['annotated_image'] = self.image_processor.annotate_violation(
                image, primary['type'], primary['confidence'], primary['bbox']
            )
        else:
            # No violation detected
            results['primary_violation'] = 'NO_VIOLATION'
            results['confidence'] = 0.9
        
        return results
    
    def batch_detect(self, images):
        """
        Detect violations in multiple images
        
        Args:
            images: List of images
        
        Returns:
            List of results
        """
        results = []
        for img in images:
            result = self.detect_violations(img)
            results.append(result)
        return results

"""
Number Plate OCR Module
Detects and extracts Indian vehicle number plates using Tesseract OCR
"""

import cv2
import numpy as np
import re
import pytesseract
from PIL import Image
import config


class NumberPlateOCR:
    """Indian Number Plate Recognition using OCR"""
    
    def __init__(self):
        """Initialize OCR with Indian plate pattern"""
        self.plate_pattern = re.compile(config.INDIAN_PLATE_PATTERN)
        # Configure pytesseract path if needed (uncomment and set path if Tesseract not in PATH)
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    def detect_plate_region(self, image):
        """
        Detect potential number plate regions in the image
        
        Args:
            image: Input image (numpy array)
        
        Returns:
            List of bounding boxes (x, y, w, h) for potential plates
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Edge detection
        edged = cv2.Canny(filtered, 30, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
        
        plate_candidates = []
        
        for contour in contours:
            # Approximate the contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
            
            # Number plates are typically rectangular(4 corners)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                
                # Check aspect ratio (Indian plates are typically 2:1 to 5:1)
                aspect_ratio = w / float(h)
                
                # Filter by aspect ratio and size
                if 1.5 < aspect_ratio < 6 and w > 50 and h > 15:
                    plate_candidates.append((x, y, w, h))
        
        return plate_candidates
    
    def preprocess_plate(self, plate_img):
        """
        Preprocess plate image for better OCR accuracy
        
        Args:
            plate_img: Cropped plate image
        
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get pure black and white
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Denoise
        denoised = cv2.medianBlur(thresh, 3)
        
        # Resize for better OCR (height of 60 pixels)
        h, w = denoised.shape
        new_height = 60
        new_width = int(w * (new_height / h))
        resized = cv2.resize(denoised, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        return resized
    
    def clean_plate_text(self, text):
        """
        Clean and format extracted text
        
        Args:
            text: Raw OCR text
        
        Returns:
            Cleaned text
        """
        # Remove special characters except spaces and hyphens
        text = re.sub(r'[^A-Z0-9\s-]', '', text.upper())
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text
    
    def validate_indian_plate(self, text):
        """
        Validate if text matches Indian number plate format
        
        Args:
            text: Extracted text
        
        Returns:
            (is_valid, formatted_text, confidence)
        """
        # Clean the text
        cleaned = self.clean_plate_text(text)
        
        # Check if it matches Indian plate pattern
        match = self.plate_pattern.search(cleaned)
        
        if match:
            plate_text = match.group(0)
            # Calculate confidence based on pattern match
            confidence = 0.9 if len(plate_text) >= 9 else 0.7
            return True, plate_text, confidence
        
        # Try alternative formats (some plates might have slight variations)
        # Example: MH12AB1234 (without spaces)
        condensed_pattern = re.compile(r'[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{1,4}')
        match = condensed_pattern.search(cleaned)
        
        if match:
            plate_text = match.group(0)
            # Add spaces for standard format
            # Format: XX ## XX ####
            formatted = f"{plate_text[:2]} {plate_text[2:4]} {plate_text[4:6]} {plate_text[6:]}"
            return True, formatted.strip(), 0.75
        
        return False, cleaned, 0.3
    
    def extract_plate(self, image):
        """
        Main method to extract number plate from image
        
        Args:
            image: Input image (numpy array or PIL Image)
        
        Returns:
            Dictionary with plate info: {
                'detected': bool,
                'plate_text': str,
                'confidence': float,
                'bbox': tuple (x, y, w, h) or None
            }
        """
        # Convert PIL to OpenCV if needed
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        result = {
            'detected': False,
            'plate_text': '',
            'confidence': 0.0,
            'bbox': None
        }
        
        # Detect plate regions
        plate_candidates = self.detect_plate_region(image)
        
        if not plate_candidates:
            return result
        
        best_confidence = 0
        best_result = None
        
        # Try OCR on each candidate
        for bbox in plate_candidates:
            x, y, w, h = bbox
            
            # Crop plate region
            plate_img = image[y:y+h, x:x+w]
            
            # Preprocess
            processed = self.preprocess_plate(plate_img)
            
            # Perform OCR
            try:
                text = pytesseract.image_to_string(
                    processed,
                    config='--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                )
                
                # Validate and clean
                is_valid, plate_text, confidence = self.validate_indian_plate(text)
                
                if is_valid and confidence > best_confidence:
                    best_confidence = confidence
                    best_result = {
                        'detected': True,
                        'plate_text': plate_text,
                        'confidence': confidence,
                        'bbox': bbox
                    }
            except Exception as e:
                print(f"OCR failed for candidate: {e}")
                continue
        
        if best_result:
            return best_result
        
        return result
    
    def extract_from_multiple(self, images):
        """
        Extract plates from multiple images
        
        Args:
            images: List of images
        
        Returns:
            List of results
        """
        results = []
        for img in images:
            result = self.extract_plate(img)
            results.append(result)
        return results

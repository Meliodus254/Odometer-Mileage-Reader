import easyocr
import cv2
import numpy as np
import re
from PIL import Image
import yaml
from typing import Tuple, Optional

class OdometerOCR:
    def __init__(self):
        config = self._load_config()
        self.reader = easyocr.Reader(['en'], gpu=True)
        self.allowed_chars = config['ocr']['allowed_characters']
        self.confidence_threshold = config['ocr']['confidence_threshold']
        
    def _load_config(self):
        with open('../config.yaml', 'r') as f:
            return yaml.safe_load(f)
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply adaptive thresholding
        processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        
        # Denoise
        processed = cv2.medianBlur(processed, 3)
        
        # Enhance contrast
        processed = cv2.convertScaleAbs(processed, alpha=1.5, beta=0)
        
        return processed
    
    def extract_digits(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        """Extract digits from image using OCR"""
        # Preprocess image
        processed = self.preprocess_image(image)
        
        # Perform OCR with character whitelist
        results = self.reader.readtext(
            processed,
            allowlist=self.allowed_chars,
            detail=1,
            paragraph=False
        )
        
        if not results:
            return None, 0.0
        
        # Filter results by confidence and combine
        valid_results = []
        total_confidence = 0
        
        for bbox, text, confidence in results:
            if confidence >= self.confidence_threshold and text.strip():
                # Keep only digits
                digits = re.sub(r'\D', '', text)
                if digits:
                    valid_results.append((digits, confidence))
                    total_confidence += confidence
        
        if not valid_results:
            return None, 0.0
        
        # Combine all valid results
        combined_text = ''.join([text for text, _ in valid_results])
        avg_confidence = total_confidence / len(valid_results)
        
        # Clean up result
        combined_text = self._clean_reading(combined_text)
        
        return combined_text, avg_confidence
    
    def _clean_reading(self, text: str) -> str:
        """Clean and validate the extracted reading"""
        # Remove non-digit characters
        cleaned = re.sub(r'\D', '', text)
        
        # Common odometer patterns
        # If length > 7, take last 7 digits (most odometers are 6-7 digits)
        if len(cleaned) > 7:
            cleaned = cleaned[-7:]
        
        # If length is reasonable (3-7 digits), return it
        if 3 <= len(cleaned) <= 7:
            return cleaned
        
        return ""
    
    def process_odometer_image(self, image_path: str, bbox: Optional[Tuple] = None) -> dict:
        """Process full odometer image"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # If bbox provided, crop to odometer region
        if bbox:
            x, y, w, h = bbox
            cropped = image[y:y+h, x:x+w]
        else:
            cropped = image
        
        # Extract reading
        reading, confidence = self.extract_digits(cropped)
        
        return {
            'reading': reading,
            'confidence': confidence,
            'success': reading is not None and len(reading) >= 3
        }

def test_ocr():
    """Test the OCR system"""
    ocr = OdometerOCR()
    
    # Test with sample images
    test_images = [
        "path/to/test/image1.jpg",
        "path/to/test/image2.jpg"
    ]
    
    for img_path in test_images:
        result = ocr.process_odometer_image(img_path)
        print(f"Image: {img_path}")
        print(f"Reading: {result['reading']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Success: {result['success']}")
        print("-" * 50)

if __name__ == '__main__':
    test_ocr()
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import pymupdf  # PyMuPDF
from pdf2image import convert_from_path
import os

class PageProcessor:
    """Handle page layout analysis and region detection"""
    
    def __init__(self):
        self.load_models()
    
    def load_models(self):
        """Load necessary models"""
        try:
            # Load YOLO for region detection (you'll need to train this)
            self.region_detector = YOLO('models/region_detector.pt')
        except:
            print("Region detector not found. Using fallback segmentation.")
            self.region_detector = None
    
    def pdf_to_images(self, pdf_path):
        """Convert PDF to images"""
        images = convert_from_path(pdf_path, dpi=300)
        return images
    
    def preprocess_image(self, image):
        """Clean and prepare image for OCR"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Deskew
        coords = np.column_stack(np.where(denoised > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        if abs(angle) > 0.5:  # Only correct if significant skew
            (h, w) = denoised.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            denoised = cv2.warpAffine(denoised, M, (w, h), 
                                    flags=cv2.INTER_CUBIC, 
                                    borderMode=cv2.BORDER_REPLICATE)
        
        # Binarize
        binary = cv2.adaptiveThreshold(denoised, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        
        return binary
    
    def detect_regions(self, image):
        """Detect question/answer regions"""
        if self.region_detector:
            # Use trained YOLO model
            results = self.region_detector(image)
            regions = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = box.conf[0].item()
                    class_id = int(box.cls[0].item())
                    
                    regions.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence,
                        'type': ['question', 'answer', 'table'][class_id]
                    })
            return regions
        else:
            # Fallback: simple contour-based segmentation
            return self.fallback_segmentation(image)
    
    def fallback_segmentation(self, image):
        """Simple contour-based region detection"""
        # Find contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                
                # Classify region type based on position and size
                region_type = 'question' if y < image.shape[0] // 2 else 'answer'
                
                regions.append({
                    'bbox': [x, y, x + w, y + h],
                    'confidence': 0.8,
                    'type': region_type
                })
        
        return regions
    
    def extract_regions(self, image, regions):
        """Extract image regions based on bounding boxes"""
        extracted_regions = []
        
        for region in regions:
            x1, y1, x2, y2 = region['bbox']
            
            # Extract region
            region_img = image[y1:y2, x1:x2]
            
            # Clean extracted region
            cleaned_region = self.preprocess_image(region_img)
            
            extracted_regions.append({
                'image': cleaned_region,
                'type': region['type'],
                'bbox': region['bbox'],
                'confidence': region['confidence']
            })
        
        return extracted_regions
    
    def process_page(self, image_path):
        """Complete page processing pipeline"""
        # Load image
        if image_path.endswith('.pdf'):
            images = self.pdf_to_images(image_path)
            image = np.array(images[0])  # Process first page
        else:
            image = cv2.imread(image_path)
        
        # Preprocess
        processed_image = self.preprocess_image(image)
        
        # Detect regions
        regions = self.detect_regions(processed_image)
        
        # Extract regions
        extracted_regions = self.extract_regions(processed_image, regions)
        
        return {
            'original_image': image,
            'processed_image': processed_image,
            'regions': extracted_regions
        }

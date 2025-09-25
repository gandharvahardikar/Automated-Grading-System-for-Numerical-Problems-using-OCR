import torch
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import cv2
import os
import sys
sys.path.append('..')
from training.character_ocr import CharacterCNN, create_character_mapping

class OCREngine:
    """Integrated OCR pipeline"""
    
    def __init__(self):
        self.load_models()
    
    def load_models(self):
        """Load all trained models"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load character model
        try:
            char_to_idx, idx_to_char, num_classes = create_character_mapping()
            self.char_model = CharacterCNN(num_classes)
            self.char_model.load_state_dict(torch.load('models/character_cnn.pth', map_location=device))
            self.char_model.eval()
            self.char_to_idx = char_to_idx
            self.idx_to_char = idx_to_char
            print("Character model loaded successfully")
        except Exception as e:
            print(f"Failed to load character model: {e}")
            self.char_model = None
        
        # Load sequence model (TrOCR)
        try:
            self.sequence_processor = TrOCRProcessor.from_pretrained('models/trocr_math')
            self.sequence_model = VisionEncoderDecoderModel.from_pretrained('models/trocr_math')
            print("Sequence model loaded successfully")
        except Exception as e:
            print(f"Failed to load sequence model, using default: {e}")
            self.sequence_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
            self.sequence_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    
    def recognize_character(self, char_image):
        """Recognize single character"""
        if self.char_model is None:
            return "?"
        
        # Preprocess
        if isinstance(char_image, np.ndarray):
            char_image = Image.fromarray(char_image)
        
        char_image = char_image.convert('L').resize((32, 32))
        char_tensor = torch.FloatTensor(np.array(char_image)).unsqueeze(0).unsqueeze(0) / 255.0
        char_tensor = (char_tensor - 0.5) / 0.5  # Normalize
        
        # Predict
        with torch.no_grad():
            output = self.char_model(char_tensor)
            predicted_idx = torch.argmax(output, dim=1).item()
        
        return self.idx_to_char.get(predicted_idx, "?")
    
    def recognize_sequence(self, sequence_image):
        """Recognize text sequence"""
        if isinstance(sequence_image, np.ndarray):
            sequence_image = Image.fromarray(sequence_image)
        
        sequence_image = sequence_image.convert('RGB')
        
        # Process with TrOCR
        pixel_values = self.sequence_processor(images=sequence_image, return_tensors="pt").pixel_values
        generated_ids = self.sequence_model.generate(pixel_values, max_length=64)
        result = self.sequence_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return result
    
    def segment_line_to_characters(self, line_image, min_char_width=8):
        """Segment line into individual characters"""
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            255 - line_image, connectivity=8
        )
        
        characters = []
        for i in range(1, num_labels):  # Skip background
            x, y, w, h, area = stats[i]
            
            # Filter by size
            if w >= min_char_width and h >= min_char_width and area > 50:
                char_image = line_image[y:y+h, x:x+w]
                characters.append({
                    'image': char_image,
                    'bbox': [x, y, x+w, y+h],
                    'area': area
                })
        
        # Sort by x-coordinate (left to right)
        characters.sort(key=lambda c: c['bbox'][0])
        
        return characters
    
    def process_region(self, region_image, region_type):
        """Process a specific region"""
        result = {
            'type': region_type,
            'text': '',
            'characters': [],
            'confidence': 0.0
        }
        
        try:
            if region_type in ['equation', 'math', 'answer']:
                # Use sequence recognition for math
                text = self.recognize_sequence(region_image)
                result['text'] = text
                result['confidence'] = 0.8
                
                # Also try character-by-character for comparison
                if self.char_model:
                    characters = self.segment_line_to_characters(region_image)
                    char_results = []
                    for char in characters:
                        recognized_char = self.recognize_character(char['image'])
                        char_results.append({
                            'character': recognized_char,
                            'bbox': char['bbox']
                        })
                    result['characters'] = char_results
                    
            else:
                # Use sequence recognition for text
                text = self.recognize_sequence(region_image)
                result['text'] = text
                result['confidence'] = 0.7
                
        except Exception as e:
            print(f"Error processing region: {e}")
            result['text'] = ""
            result['confidence'] = 0.0
        
        return result

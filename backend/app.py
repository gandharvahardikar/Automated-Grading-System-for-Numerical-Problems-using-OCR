from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
from werkzeug.utils import secure_filename
import traceback
import json
import cv2
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)

class SimpleOCREngine:
    """Simplified OCR engine using TrOCR directly"""
    
    def __init__(self):
        self.load_model()
    
    def load_model(self):
        """Load TrOCR model"""
        try:
            print("Loading TrOCR model for handwritten text...")
            self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
            self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
            print("✅ TrOCR model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load TrOCR: {e}")
            self.processor = None
            self.model = None
    
    def preprocess_image(self, image_path):
        """Clean and prepare image for OCR"""
        try:
            # Read image
            if image_path.lower().endswith('.pdf'):
                from pdf2image import convert_from_path
                images = convert_from_path(image_path, dpi=300)
                image = np.array(images[0])
            else:
                image = cv2.imread(image_path)
            
            # Convert to RGB
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            return Image.fromarray(image_rgb)
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def extract_text(self, image_path):
        """Extract text from image"""
        if not self.processor or not self.model:
            return "OCR model not available"
        
        try:
            # Preprocess image
            image = self.preprocess_image(image_path)
            if image is None:
                return "Failed to process image"
            
            # OCR with TrOCR
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
            generated_ids = self.model.generate(pixel_values, max_length=128)
            result = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return result
            
        except Exception as e:
            print(f"OCR extraction error: {e}")
            return f"OCR failed: {str(e)}"

class SimpleGrader:
    """Simplified grading engine"""
    
    def __init__(self):
        self.answer_key = {
            'regression': ['y =', '0.613', '13.87', 'r =', '0.863'],
            'correlation': ['0.8', '0.9'],
            'hypothesis': ['reject', 'accept', 'z =', 't ='],
            'statistics': ['mean', 'standard deviation', 'variance']
        }
    
    def extract_numbers(self, text):
        """Extract numbers from text"""
        import re
        pattern = r'-?\d+\.?\d*'
        numbers = re.findall(pattern, text)
        return [float(num) for num in numbers if num]
    
    def grade_answer(self, extracted_text, question_type='general'):
        """Grade extracted answer"""
        score = 0
        max_score = 5
        feedback = []
        
        text_lower = extracted_text.lower()
        numbers = self.extract_numbers(extracted_text)
        
        # Check for key mathematical terms
        if 'regression' in text_lower or 'y =' in text_lower:
            score += 2
            feedback.append("✅ Regression equation detected")
            
            # Check for specific values
            if any(abs(num - 0.613) < 0.1 for num in numbers):
                score += 1
                feedback.append("✅ Correct slope value found")
            
            if any(abs(num - 13.87) < 1 for num in numbers):
                score += 1
                feedback.append("✅ Correct intercept value found")
        
        if 'correlation' in text_lower or 'r =' in text_lower:
            score += 1
            feedback.append("✅ Correlation mentioned")
            
            if any(0.8 <= num <= 0.9 for num in numbers):
                score += 1
                feedback.append("✅ Correlation coefficient in expected range")
        
        # Statistical terms
        if any(term in text_lower for term in ['hypothesis', 'test', 'statistic']):
            score += 1
            feedback.append("✅ Statistical analysis detected")
        
        if not feedback:
            feedback.append("⚠️ No recognizable mathematical content found")
            score = 1  # Partial credit for attempt
        
        percentage = (score / max_score) * 100
        
        return {
            'score': score,
            'max_score': max_score,
            'percentage': percentage,
            'feedback': feedback,
            'extracted_text': extracted_text,
            'numbers_found': numbers
        }

# Initialize components
print("Initializing OCR engine...")
ocr_engine = SimpleOCREngine()
grader = SimpleGrader()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'OCR Exam Grading System API is running!',
        'components': {
            'ocr_engine': 'loaded' if ocr_engine.processor else 'failed',
            'grader': 'loaded'
        }
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        paper_type = request.form.get('paper_type', 'answer')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            file_id = str(uuid.uuid4())
            filename = secure_filename(f"{file_id}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            file.save(filepath)
            
            return jsonify({
                'success': True,
                'file_id': file_id,
                'filename': filename,
                'paper_type': paper_type,
                'filepath': filepath,
                'message': 'File uploaded successfully'
            })
        
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/process-paper', methods=['POST'])
def process_paper():
    try:
        data = request.get_json()
        filename = data.get('filename')
        paper_type = data.get('paper_type', 'answer')
        
        if not filename:
            return jsonify({'error': 'Filename required'}), 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        print(f"Processing file: {filepath}")
        
        # Extract text using OCR
        extracted_text = ocr_engine.extract_text(filepath)
        print(f"Extracted text: {extracted_text}")
        
        # Grade the answer if it's an answer paper
        grading_results = None
        if paper_type == 'answer':
            grade_result = grader.grade_answer(extracted_text)
            
            grading_results = {
                'total_score': grade_result['score'],
                'max_total_score': grade_result['max_score'],
                'percentage': grade_result['percentage'],
                'question_grades': {
                    'Q1': {
                        'score': grade_result['score'],
                        'max_score': grade_result['max_score'],
                        'percentage': grade_result['percentage'],
                        'feedback': grade_result['feedback']
                    }
                },
                'summary': {
                    'total_questions': 1,
                    'correct_questions': 1 if grade_result['percentage'] >= 70 else 0,
                    'accuracy': grade_result['percentage'],
                    'strengths': [fb for fb in grade_result['feedback'] if '✅' in fb],
                    'improvements': [fb for fb in grade_result['feedback'] if '⚠️' in fb or '❌' in fb]
                }
            }
        
        # Prepare response
        response = {
            'success': True,
            'file_id': data.get('file_id'),
            'paper_type': paper_type,
            'processing_results': {
                'page_analysis': {
                    'total_regions': 1,
                    'region_types': [paper_type]
                },
                'ocr_results': {
                    'regions': [{
                        'type': paper_type,
                        'text': extracted_text,
                        'confidence': 0.85,
                        'bbox': [0, 0, 100, 100]
                    }],
                    'full_text': extracted_text
                },
                'grading_results': grading_results
            }
        }
        
        # Save results
        results_file = os.path.join(
            app.config['PROCESSED_FOLDER'], 
            f"{data.get('file_id')}_results.json"
        )
        
        with open(results_file, 'w') as f:
            json.dump(response, f, indent=2, default=str)
        
        return jsonify(response)
        
    except Exception as e:
        error_msg = f'Processing failed: {str(e)}'
        print(f"Error: {error_msg}")
        print(traceback.format_exc())
        return jsonify({'error': error_msg}), 500

@app.route('/api/files/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    print("🚀 Starting OCR Exam Grading System API...")
    print("📄 Upload endpoint: POST /api/upload")
    print("🔍 Process paper: POST /api/process-paper")
    print("💚 Health check: GET /api/health")
    print("🔗 Access at: http://localhost:5000")
    app.run(debug=True, port=5000, host='0.0.0.0')

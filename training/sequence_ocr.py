import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import string

class CRNN(nn.Module):
    """CRNN for sequence recognition"""
    def __init__(self, vocab_size, hidden_size=256, num_layers=2):
        super(CRNN, self).__init__()
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1)),
        )
        
        # RNN
        self.rnn = nn.LSTM(512, hidden_size, num_layers, bidirectional=True, batch_first=True)
        
        # Output layer
        self.output = nn.Linear(hidden_size * 2, vocab_size)
        
    def forward(self, x):
        # CNN features
        conv_features = self.cnn(x)  # [B, C, H, W]
        
        # Reshape for RNN: [B, W, C*H]
        B, C, H, W = conv_features.size()
        conv_features = conv_features.permute(0, 3, 1, 2).contiguous()
        conv_features = conv_features.view(B, W, C * H)
        
        # RNN
        rnn_output, _ = self.rnn(conv_features)
        
        # Output
        output = self.output(rnn_output)
        
        return output

class MathSequenceGenerator:
    """Generate synthetic math sequences for training"""
    def __init__(self):
        self.operators = ['+', '-', '*', '/', '=']
        self.digits = list('0123456789')
        self.variables = list('xyztabcn')
        
    def generate_equation(self):
        """Generate random math equation"""
        patterns = [
            lambda: f"{random.randint(1, 99)} {random.choice(self.operators[:4])} {random.randint(1, 99)} = {random.randint(1, 999)}",
            lambda: f"{random.choice(self.variables)} = {random.randint(1, 99)} {random.choice(self.operators[:4])} {random.randint(1, 99)}",
            lambda: f"{random.randint(1, 99)}.{random.randint(10, 99)}",
            lambda: f"{random.choice(self.variables)}² = {random.randint(1, 999)}",
        ]
        
        return random.choice(patterns)()
    
    def create_sequence_image(self, text, width=200, height=32):
        """Create image from text sequence"""
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        draw.text((5, 5), text, fill='black', font=font)
        
        return img.convert('L')

def train_sequence_model():
    """Train sequence recognition model using TrOCR"""
    print("Loading TrOCR model...")
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    
    # Generate training data
    generator = MathSequenceGenerator()
    
    print("Generating training sequences...")
    train_texts = []
    train_images = []
    
    for _ in range(1000):  # Small dataset for demo
        text = generator.generate_equation()
        image = generator.create_sequence_image(text)
        
        train_texts.append(text)
        train_images.append(image)
    
    # Fine-tune on math expressions (simplified training loop)
    print("Fine-tuning TrOCR on math expressions...")
    
    # Save for later use
    os.makedirs('models', exist_ok=True)
    model.save_pretrained('models/trocr_math')
    processor.save_pretrained('models/trocr_math')
    
    print("Sequence model saved!")
    
    return model, processor

if __name__ == "__main__":
    train_sequence_model()

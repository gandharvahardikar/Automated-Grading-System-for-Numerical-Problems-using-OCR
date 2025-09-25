import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class CharacterDataset(Dataset):
    """Dataset for individual characters and symbols"""
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class CharacterCNN(nn.Module):
    """CNN for character recognition"""
    def __init__(self, num_classes=72):  # 0-9, A-Z, a-z, +, -, *, /, =, (, ), etc.
        super(CharacterCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def create_character_mapping():
    """Create mapping for characters to indices"""
    chars = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
    symbols = ['+', '-', '*', '/', '=', '(', ')', '.', ',', '%', '²', '³', '√', 'π', 'σ', 'μ', 'α', 'β']
    all_chars = chars + symbols
    
    char_to_idx = {char: idx for idx, char in enumerate(all_chars)}
    idx_to_char = {idx: char for idx, char in enumerate(all_chars)}
    
    return char_to_idx, idx_to_char, len(all_chars)

def generate_synthetic_data(num_samples=10000):
    """Generate synthetic character images for training"""
    from PIL import Image, ImageDraw, ImageFont
    import random
    
    char_to_idx, _, num_classes = create_character_mapping()
    images = []
    labels = []
    
    # Create synthetic images
    for _ in range(num_samples):
        # Random character
        char = random.choice(list(char_to_idx.keys()))
        label = char_to_idx[char]
        
        # Create image
        img = Image.new('RGB', (32, 32), color='white')
        draw = ImageDraw.Draw(img)
        
        # Add some noise and variation
        font_size = random.randint(16, 24)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Random position
        x = random.randint(2, 10)
        y = random.randint(2, 10)
        
        # Draw character
        draw.text((x, y), char, fill='black', font=font)
        
        # Convert to grayscale and numpy
        img = img.convert('L')
        img_array = np.array(img)
        
        images.append(img_array)
        labels.append(label)
    
    return np.array(images), np.array(labels)

def train_character_model():
    """Train character recognition model"""
    print("Generating synthetic training data...")
    images, labels = generate_synthetic_data(20000)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Datasets
    train_dataset = CharacterDataset(X_train, y_train, transform)
    test_dataset = CharacterDataset(X_test, y_test, transform)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Model, loss, optimizer
    _, _, num_classes = create_character_mapping()
    model = CharacterCNN(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Training loop
    print("Starting training...")
    model.train()
    for epoch in range(20):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}')
    
    # Test accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/character_cnn.pth')
    
    # Save mappings
    char_to_idx, idx_to_char, _ = create_character_mapping()
    torch.save({'char_to_idx': char_to_idx, 'idx_to_char': idx_to_char}, 'models/char_mappings.pth')
    
    print("Model saved successfully!")
    return model

if __name__ == "__main__":
    train_character_model()

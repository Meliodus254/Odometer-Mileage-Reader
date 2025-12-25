# classification/train_classification_fixed.py - COMPLETELY FIXED VERSION
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
import yaml
import os
import sys
import json
import cv2
from sklearn.model_selection import train_test_split
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Custom Dataset class - FIXED VERSION
class OdometerDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            # Load with PIL for better compatibility
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            else:
                # Default transform if none provided
                image = transforms.ToTensor()(image)
                
            return image, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            print(f"Warning: Could not load image {image_path}: {e}")
            # Return dummy data of correct shape
            dummy_image = torch.zeros((3, 256, 256), dtype=torch.float32)
            return dummy_image, torch.tensor(label, dtype=torch.long)

# Define the model
class TransferLearningModel(nn.Module):
    """Using pretrained ResNet50 for better accuracy"""
    def __init__(self, num_classes=2):
        super(TransferLearningModel, self).__init__()
        
        try:
            # Load pretrained ResNet50
            self.model = models.resnet50(pretrained=True)
            
            # Freeze early layers
            for param in list(self.model.parameters())[:-20]:
                param.requires_grad = False
            
            # Replace the final layer
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
            print("[OK] Using pretrained ResNet50")
        except Exception as e:
            print(f"[WARNING] ResNet50 not available ({e}), using simple CNN")
            self.model = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, num_classes)
            )
    
    def forward(self, x):
        return self.model(x)

def load_config():
    """Load configuration with proper path resolution"""
    # Get the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, 'config.yaml')
    
    if not os.path.exists(config_path):
        print(f"[WARNING] Config file not found at {config_path}, using defaults")
        return {
            'paths': {
                'images_dir': os.path.join(project_root, 'trodo-v01/images'),
                'ground_truth': os.path.join(project_root, 'trodo-v01/ground-truth/groundtruth.json')
            },
            'classification': {
                'batch_size': 32,
                'epochs': 30,
                'learning_rate': 0.001,
                'image_size': [256, 256],
                'test_size': 0.2
            }
        }
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure paths are absolute
    if 'paths' in config:
        # Handle images_dir
        if 'images_dir' in config['paths']:
            if not os.path.isabs(config['paths']['images_dir']):
                config['paths']['images_dir'] = os.path.join(project_root, config['paths']['images_dir'])
        
        # Handle ground_truth
        if 'ground_truth' in config['paths']:
            if not os.path.isabs(config['paths']['ground_truth']):
                config['paths']['ground_truth'] = os.path.join(project_root, config['paths']['ground_truth'])
            
            # Check if file exists, try alternative locations
            if not os.path.exists(config['paths']['ground_truth']):
                print(f"[WARNING] Ground truth not found at: {config['paths']['ground_truth']}")
                
                # Try alternative locations
                alt_paths = [
                    os.path.join(project_root, 'groundtruth.json'),
                    os.path.join(project_root, 'trodo-v01', 'groundtruth.json'),
                    os.path.join(project_root, 'ground-truth', 'groundtruth.json')
                ]
                
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        config['paths']['ground_truth'] = alt_path
                        print(f"[OK] Found ground truth at: {alt_path}")
                        break
                else:
                    print("[ERROR] Could not find ground truth file")
    
    return config

def load_classification_data(config):
    """Load and prepare classification data with error handling"""
    LABEL_MAP = {'analog': 0, 'digital': 1}
    
    images_dir = config['paths']['images_dir']
    ground_truth_path = config['paths']['ground_truth']
    image_size = tuple(config['classification']['image_size'])
    test_size = config['classification']['test_size']
    
    print(f"\n[INFO] Loading data from:")
    print(f"   Images: {images_dir}")
    print(f"   Ground truth: {ground_truth_path}")
    print(f"   Image size: {image_size}")
    
    # Check if paths exist
    if not os.path.exists(images_dir):
        print(f"[ERROR] Images directory not found: {images_dir}")
        
        # Try to find images in alternative locations
        alt_paths = [
            os.path.join(os.path.dirname(images_dir), 'images'),
            os.path.join(project_root, 'trodo-v01', 'images'),
            os.path.join(project_root, 'images')
        ]
        
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                images_dir = alt_path
                print(f"[OK] Found images at: {alt_path}")
                break
        else:
            print("[ERROR] Could not find images directory")
            return None
    
    if not os.path.exists(ground_truth_path):
        print(f"[ERROR] Ground truth file not found: {ground_truth_path}")
        return None
    
    try:
        with open(ground_truth_path, 'r') as file:
            data = json.load(file)
        
        # Handle different JSON structures
        if 'odometers' in data:
            odometers = data['odometers']
        elif isinstance(data, list):
            odometers = data
        else:
            print("[ERROR] Unexpected JSON structure")
            return None
            
        print(f"[OK] Loaded {len(odometers)} samples from ground truth")
    except Exception as e:
        print(f"[ERROR] Error loading ground truth: {e}")
        return None
    
    image_paths = []
    labels = []
    
    successful_loads = 0
    failed_loads = 0
    
    for i, item in enumerate(odometers):
        try:
            # Get image filename
            if 'image' in item:
                image_filename = item['image']
            elif 'filename' in item:
                image_filename = item['filename']
            else:
                print(f"[WARNING] Item {i} has no image field: {item}")
                failed_loads += 1
                continue
            
            # Construct image path
            image_path = os.path.join(images_dir, image_filename)
            
            # Try alternative image paths if not found
            if not os.path.exists(image_path):
                # Try with different extensions
                basename = os.path.splitext(image_filename)[0]
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    alt_path = os.path.join(images_dir, basename + ext)
                    if os.path.exists(alt_path):
                        image_path = alt_path
                        break
            
            if not os.path.exists(image_path):
                if failed_loads < 5:  # Only show first few failures
                    print(f"[WARNING] Image not found: {image_filename}")
                failed_loads += 1
                continue
            
            # Check if we can read the image
            try:
                # Try to open the image to verify it's valid
                img = Image.open(image_path)
                img.verify()  # Verify it's a valid image file
                img.close()
                
                image_paths.append(image_path)
                
                # Get label
                if 'odometer_type' in item:
                    label = item['odometer_type'].lower()
                    if label in LABEL_MAP:
                        labels.append(LABEL_MAP[label])
                    else:
                        print(f"[WARNING] Unknown label '{label}' for image {image_filename}")
                        labels.append(0)  # Default to analog
                else:
                    labels.append(0)  # Default to analog
                
                successful_loads += 1
                
                if successful_loads % 500 == 0:
                    print(f"   Processed {successful_loads} images...")
                    
            except Exception as e:
                print(f"[WARNING] Invalid image file {image_path}: {e}")
                failed_loads += 1
                continue
                
        except Exception as e:
            print(f"[WARNING] Error processing item {i}: {e}")
            failed_loads += 1
            continue
    
    if len(image_paths) == 0:
        print("[ERROR] No images were loaded successfully")
        return None
    
    print(f"\n[INFO] Data loading summary:")
    print(f"   [OK] Successfully loaded: {successful_loads} images")
    print(f"   [ERROR] Failed to load: {failed_loads} images")
    print(f"   [INFO] Class distribution:")
    print(f"      - Analog: {sum(l == 0 for l in labels)} ({sum(l == 0 for l in labels)/len(labels)*100:.1f}%)")
    print(f"      - Digital: {sum(l == 1 for l in labels)} ({sum(l == 1 for l in labels)/len(labels)*100:.1f}%)")
    
    # Split the data
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    return train_paths, train_labels, test_paths, test_labels

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cuda'):
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_acc = 0.0
    best_model_state = None
    
    # Ensure models directory exists
    models_dir = os.path.join(project_root, 'models', 'classification')
    os.makedirs(models_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{total_batches}, Loss: {loss.item():.4f}")
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = 100 * correct / total
        
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {epoch_train_loss:.4f}')
        print(f'  Val Loss: {epoch_val_loss:.4f}')
        print(f'  Val Acc: {epoch_val_acc:.2f}%')
        
        # Save best model
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_state = model.state_dict().copy()
            model_path = os.path.join(models_dir, 'best_classification_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_acc,
                'val_loss': epoch_val_loss,
            }, model_path)
            print(f"  [SAVED] New best model with accuracy: {best_acc:.2f}%")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Validation Accuracy')
    
    plt.tight_layout()
    history_path = os.path.join(models_dir, 'training_history.png')
    plt.savefig(history_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved training history to {history_path}")
    
    return best_model_state, best_acc

def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Generate classification report
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    report = classification_report(all_labels, all_preds, target_names=['Analog', 'Digital'])
    print(report)
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Analog', 'Digital'],
                yticklabels=['Analog', 'Digital'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    cm_path = os.path.join(project_root, 'models', 'classification', 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved confusion matrix to {cm_path}")
    
    accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f"[OK] Overall Accuracy: {accuracy:.2f}%")
    
    return accuracy

def main():
    print("\n" + "="*60)
    print("STARTING CLASSIFICATION TRAINING")
    print("="*60)
    
    # Load config
    config = load_config()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[INFO] Using device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    print("\n[INFO] Loading datasets...")
    data = load_classification_data(config)
    
    if data is None:
        print("[ERROR] Failed to load data. Exiting.")
        return
    
    train_paths, train_labels, test_paths, test_labels = data
    
    # Split test into validation and test
    test_paths, val_paths, test_labels, val_labels = train_test_split(
        test_paths, test_labels, test_size=0.5, random_state=42
    )
    
    print(f"\n[INFO] Dataset sizes:")
    print(f"   Train: {len(train_paths)} samples")
    print(f"   Val: {len(val_paths)} samples")
    print(f"   Test: {len(test_paths)} samples")
    
    # Create datasets with augmentation for training
    image_size = tuple(config['classification']['image_size'])
    
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = OdometerDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = OdometerDataset(val_paths, val_labels, transform=val_transform)
    test_dataset = OdometerDataset(test_paths, test_labels, transform=val_transform)
    
    # Create data loaders
    batch_size = config['classification'].get('batch_size', 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)
    
    # Initialize model
    print("\n[INFO] Initializing model...")
    model = TransferLearningModel(num_classes=2).to(device)
    
    # Print model architecture
    print("\n[INFO] Model summary:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    learning_rate = config['classification'].get('learning_rate', 0.001)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                    factor=0.5, patience=5, 
                                                    verbose=True)
    
    # Train model
    print("\n" + "="*50)
    print("STARTING TRAINING")
    print("="*50)
    
    num_epochs = config['classification'].get('epochs', 30)
    best_model_state, best_acc = train_model(model, train_loader, val_loader, criterion, optimizer,
                                           num_epochs=num_epochs, device=device)
    
    # Load best model and evaluate on test set
    model_path = os.path.join(project_root, 'models', 'classification', 'best_classification_model.pth')
    if os.path.exists(model_path):
        print(f"\n[INFO] Loading best model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"   Best validation accuracy: {checkpoint['val_acc']:.2f}%")
        print(f"   Best validation loss: {checkpoint['val_loss']:.4f}")
        
        test_accuracy = evaluate_model(model, test_loader, device)
        
        print(f"\n" + "="*50)
        print("TRAINING RESULTS")
        print("="*50)
        print(f"   Best Validation Accuracy: {checkpoint['val_acc']:.2f}%")
        print(f"   Test Accuracy: {test_accuracy:.2f}%")
        
        if test_accuracy > 90:
            print("\n[SUCCESS] Model achieved above 90% accuracy!")
        elif test_accuracy > 80:
            print("\n[GOOD] Model accuracy between 80-90%")
        elif test_accuracy > 70:
            print("\n[FAIR] Model accuracy between 70-80%")
        else:
            print("\n[NEEDS IMPROVEMENT] Model accuracy below 70%, consider more training or tuning")
    else:
        print(f"\n[WARNING] Model file not found: {model_path}")
        print("Using the last trained model for evaluation...")
        test_accuracy = evaluate_model(model, test_loader, device)
    
    print("\n" + "="*60)
    print("CLASSIFICATION TRAINING COMPLETE")
    print("="*60)
    print(f"\n[INFO] Models saved to: {os.path.join(project_root, 'models', 'classification')}")
    print(f"[INFO] Visualizations saved to: {os.path.join(project_root, 'models', 'classification')}")

if __name__ == '__main__':
    main()
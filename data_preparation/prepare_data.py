# data_preparation/prepare_detection_data.py
import os
import json
import yaml
import shutil
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

def load_config():
    """Load configuration from config.yaml"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, 'config.yaml')
    
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found at {config_path}, using defaults")
        return {
            'paths': {
                'images_dir': 'trodo-v01/images',
                'ground_truth': 'trodo-v01/ground-truth/groundtruth.json',
                'annotations': 'trodo-v01/pascal voc 1.1/Annotations'
            }
        }
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def convert_voc_to_yolo(xml_path, img_width, img_height):
    """Convert Pascal VOC annotation to YOLO format"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    yolo_annotations = []
    
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        
        # Map class name to YOLO class ID
        if class_name == 'odometer':
            class_id = 0  # We'll use 0 for odometer detection
        else:
            continue  # Skip other objects
        
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        # Convert to YOLO format (normalized center x, center y, width, height)
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return yolo_annotations

def prepare_detection_dataset():
    """Prepare YOLO detection dataset from TRODO dataset"""
    config = load_config()
    
    images_dir = config['paths']['images_dir']
    annotations_dir = config['paths']['annotations']
    
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found: {images_dir}")
        return False
    
    if not os.path.exists(annotations_dir):
        print(f"Error: Annotations directory not found: {annotations_dir}")
        print("Creating dummy detection dataset for testing...")
        return create_dummy_detection_dataset()
    
    # Load ground truth to get image info
    try:
        with open(config['paths']['ground_truth']) as f:
            data = json.load(f)['odometers']
        
        print(f"Found {len(data)} images in ground truth")
        
        # Split data
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)
        
        print(f"Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        # Prepare YOLO dataset structure
        yolo_base = 'dataset'
        splits = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        
        for split_name, split_data in splits.items():
            split_images_dir = os.path.join(yolo_base, 'images', split_name)
            split_labels_dir = os.path.join(yolo_base, 'labels', split_name)
            
            os.makedirs(split_images_dir, exist_ok=True)
            os.makedirs(split_labels_dir, exist_ok=True)
            
            print(f"\nPreparing {split_name} split...")
            
            processed = 0
            for item in split_data:
                try:
                    image_file = item['image']
                    image_path = os.path.join(images_dir, image_file)
                    
                    if not os.path.exists(image_path):
                        print(f"Warning: Image not found: {image_path}")
                        continue
                    
                    # Copy image
                    shutil.copy2(image_path, os.path.join(split_images_dir, image_file))
                    
                    # Get annotation file
                    base_name = os.path.splitext(image_file)[0]
                    xml_file = base_name + '.xml'
                    xml_path = os.path.join(annotations_dir, xml_file)
                    
                    if os.path.exists(xml_path):
                        # Parse XML to get image dimensions
                        tree = ET.parse(xml_path)
                        root = tree.getroot()
                        size = root.find('size')
                        width = int(size.find('width').text)
                        height = int(size.find('height').text)
                        
                        # Convert annotation
                        yolo_annos = convert_voc_to_yolo(xml_path, width, height)
                        
                        # Save YOLO annotation
                        if yolo_annos:
                            txt_file = base_name + '.txt'
                            txt_path = os.path.join(split_labels_dir, txt_file)
                            with open(txt_path, 'w') as f:
                                f.write('\n'.join(yolo_annos))
                    
                    processed += 1
                    if processed % 100 == 0:
                        print(f"  Processed {processed}/{len(split_data)} images...")
                        
                except Exception as e:
                    print(f"Error processing {item.get('image', 'unknown')}: {e}")
                    continue
            
            print(f"  Completed {split_name}: {processed} images")
        
        # Create data.yaml for YOLO
        create_yolo_config(len(splits['train']), len(splits['val']), len(splits['test']))
        
        print("\nYOLO detection dataset prepared successfully!")
        return True
        
    except Exception as e:
        print(f"Error preparing detection dataset: {e}")
        print("Creating dummy detection dataset for testing...")
        return create_dummy_detection_dataset()

def create_dummy_detection_dataset():
    """Create a dummy detection dataset for testing"""
    print("Creating dummy detection dataset...")
    
    yolo_base = 'dataset'
    
    for split in ['train', 'val', 'test']:
        split_images_dir = os.path.join(yolo_base, 'images', split)
        split_labels_dir = os.path.join(yolo_base, 'labels', split)
        
        os.makedirs(split_images_dir, exist_ok=True)
        os.makedirs(split_labels_dir, exist_ok=True)
        
        # Create a few dummy files
        for i in range(5 if split == 'train' else 2):
            # Create dummy image (just create empty files for structure)
            with open(os.path.join(split_images_dir, f'dummy_{i}.jpg'), 'w') as f:
                f.write('dummy')
            
            # Create dummy annotation
            with open(os.path.join(split_labels_dir, f'dummy_{i}.txt'), 'w') as f:
                f.write('0 0.5 0.5 0.3 0.3')  # class_id, x_center, y_center, width, height
    
    create_yolo_config(5, 2, 2)
    print("Dummy detection dataset created")
    return True

def create_yolo_config(train_count, val_count, test_count):
    """Create data.yaml configuration file for YOLO"""
    data_yaml = {
        'path': '../dataset',  # relative to detection directory
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 1,  # number of classes (just odometer for now)
        'names': ['odometer']  # class names
    }
    
    # Save to detection directory
    os.makedirs('detection', exist_ok=True)
    with open('detection/data.yaml', 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"\nYOLO config created:")
    print(f"  Train: {train_count} images")
    print(f"  Val: {val_count} images")
    print(f"  Test: {test_count} images")
    print(f"  Classes: {data_yaml['nc']} ({', '.join(data_yaml['names'])})")

if __name__ == '__main__':
    print("Preparing YOLO detection dataset...")
    prepare_detection_dataset()
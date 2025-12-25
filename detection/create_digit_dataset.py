# detection/create_digit_dataset.py
import os
import sys
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def main():
    print("\n" + "="*60)
    print("DIGIT DATASET CREATOR (Using XML Annotations)")
    print("="*60)
    
    # Paths
    project_root = Path(__file__).parent.parent
    raw_data_dir = project_root / "trodo-v01"
    output_dir = project_root / "dataset_digits"
    
    print(f"📁 Project root: {project_root}")
    print(f"📁 Raw data: {raw_data_dir}")
    print(f"📁 Output: {output_dir}")
    
    if not raw_data_dir.exists():
        print(f"❌ Raw data directory not found: {raw_data_dir}")
        return
    
    # Create output directories
    output_dirs = {
        'images': {
            'train': output_dir / 'images' / 'train',
            'val': output_dir / 'images' / 'val',
            'test': output_dir / 'images' / 'test'
        },
        'labels': {
            'train': output_dir / 'labels' / 'train',
            'val': output_dir / 'labels' / 'val',
            'test': output_dir / 'labels' / 'test'
        }
    }
    
    for split in ['train', 'val', 'test']:
        output_dirs['images'][split].mkdir(parents=True, exist_ok=True)
        output_dirs['labels'][split].mkdir(parents=True, exist_ok=True)
    
    # Paths to raw data
    annotations_dir = raw_data_dir / "pascal voc 1.1" / "annotations"
    images_dir = raw_data_dir / "images"
    groundtruth_path = raw_data_dir / "ground-truth" / "groundtruth.json"
    
    if not annotations_dir.exists():
        print(f"❌ Annotations directory not found: {annotations_dir}")
        return
    
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return
    
    # Load ground truth for mileage values
    groundtruth = {}
    if groundtruth_path.exists():
        with open(groundtruth_path, 'r') as f:
            gt_data = json.load(f)
            groundtruth = {v["image"]: v["mileage"] for v in gt_data.get("odometers", [])}
        print(f"✅ Loaded {len(groundtruth)} ground truth mileage values")
    
    # Get all XML files
    xml_files = list(annotations_dir.glob("*.xml"))
    random.shuffle(xml_files)
    
    print(f"\n📊 Found {len(xml_files)} annotation files")
    
    # Split dataset
    total = len(xml_files)
    test_count = max(1, int(total * 0.1))  # 10% test
    val_count = max(1, int(total * 0.15))  # 15% validation
    train_count = total - test_count - val_count
    
    print(f"📈 Dataset split:")
    print(f"  Training: {train_count} samples")
    print(f"  Validation: {val_count} samples")
    print(f"  Test: {test_count} samples")
    
    # Create splits
    test_files = set(xml_files[:test_count])
    val_files = set(xml_files[test_count:test_count + val_count])
    train_files = set(xml_files[test_count + val_count:])
    
    # Process each annotation
    digit_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    class_map = {digit: idx for idx, digit in enumerate(digit_classes)}
    
    stats = {
        'train': {'images': 0, 'digits': 0},
        'val': {'images': 0, 'digits': 0},
        'test': {'images': 0, 'digits': 0}
    }
    
    skipped_images = []
    
    print("\n🚀 Processing images...")
    
    for xml_file in xml_files:
        image_name = xml_file.stem + ".jpg"
        image_path = images_dir / image_name
        
        if not image_path.exists():
            print(f"⚠️ Image not found: {image_name}")
            skipped_images.append(image_name)
            continue
        
        # Determine split
        if xml_file in test_files:
            split = 'test'
        elif xml_file in val_files:
            split = 'val'
        else:
            split = 'train'
        
        # Load image
        try:
            img = Image.open(image_path).convert('RGB')
            img_width, img_height = img.size
        except Exception as e:
            print(f"❌ Failed to load image {image_name}: {e}")
            skipped_images.append(image_name)
            continue
        
        # Parse XML to get digit annotations
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Find odometer box from XML
            odometer_box = None
            for obj in root.findall("object"):
                name = obj.find("name").text
                if name == "odometer":
                    bndbox = obj.find("bndbox")
                    xmin = int(float(bndbox.find("xmin").text))
                    ymin = int(float(bndbox.find("ymin").text))
                    xmax = int(float(bndbox.find("xmax").text))
                    ymax = int(float(bndbox.find("ymax").text))
                    odometer_box = (xmin, ymin, xmax, ymax)
                    break
            
            if not odometer_box:
                print(f"⚠️ No odometer annotation found in {image_name}")
                skipped_images.append(image_name)
                continue
            
            # Crop odometer region
            try:
                cropped_img = img.crop(odometer_box)
                crop_width, crop_height = cropped_img.size
                
                # Save cropped image
                output_image_path = output_dirs['images'][split] / image_name
                cropped_img.save(output_image_path)
                
                # Collect digit annotations
                digit_annotations = []
                for obj in root.findall("object"):
                    name = obj.find("name").text
                    if name in digit_classes:
                        bndbox = obj.find("bndbox")
                        xmin = int(float(bndbox.find("xmin").text))
                        ymin = int(float(bndbox.find("ymin").text))
                        xmax = int(float(bndbox.find("xmax").text))
                        ymax = int(float(bndbox.find("ymax").text))
                        
                        # Convert to cropped coordinates
                        x0, y0 = odometer_box[0], odometer_box[1]
                        xmin_crop = xmin - x0
                        ymin_crop = ymin - y0
                        xmax_crop = xmax - x0
                        ymax_crop = ymax - y0
                        
                        # Skip if digit is outside cropped region
                        if (xmin_crop < 0 or ymin_crop < 0 or 
                            xmax_crop > crop_width or ymax_crop > crop_height):
                            continue
                        
                        # Convert to YOLO format
                        x_center = ((xmin_crop + xmax_crop) / 2) / crop_width
                        y_center = ((ymin_crop + ymax_crop) / 2) / crop_height
                        width = (xmax_crop - xmin_crop) / crop_width
                        height = (ymax_crop - ymin_crop) / crop_height
                        
                        digit_annotations.append(
                            f"{class_map[name]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                        )
                
                # Save labels
                if digit_annotations:
                    output_label_path = output_dirs['labels'][split] / (xml_file.stem + ".txt")
                    with open(output_label_path, 'w') as f:
                        f.write('\n'.join(digit_annotations))
                    
                    stats[split]['images'] += 1
                    stats[split]['digits'] += len(digit_annotations)
                    print(f"✅ {split}: {image_name} - {len(digit_annotations)} digits")
                else:
                    print(f"⚠️ {split}: {image_name} - No digits found in cropped region")
                    skipped_images.append(image_name)
                
            except Exception as e:
                print(f"❌ Error cropping {image_name}: {e}")
                skipped_images.append(image_name)
                
        except Exception as e:
            print(f"❌ Error parsing XML {xml_file.name}: {e}")
            skipped_images.append(image_name)
    
    # Create YAML configuration file - FIXED: no backslash in f-string
    path_str = str(output_dir).replace('\\', '/')
    yaml_content = f"""path: {path_str}
train: images/train
val: images/val

nc: {len(digit_classes)}
names: {digit_classes}
"""
    
    yaml_path = project_root / "detection" / "trodo_stage2.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    # Print statistics
    print("\n" + "="*60)
    print("DATASET CREATION COMPLETE")
    print("="*60)
    
    total_images = sum(stats[split]['images'] for split in stats)
    total_digits = sum(stats[split]['digits'] for split in stats)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total images: {total_images}")
    print(f"  Total digit instances: {total_digits}")
    print(f"  Average digits per image: {total_digits/total_images if total_images > 0 else 0:.2f}")
    
    for split in ['train', 'val', 'test']:
        print(f"\n  {split.upper()}:")
        print(f"    Images: {stats[split]['images']}")
        print(f"    Digits: {stats[split]['digits']}")
        if stats[split]['images'] > 0:
            print(f"    Avg per image: {stats[split]['digits']/stats[split]['images']:.2f}")
    
    print(f"\n✅ YAML configuration saved to: {yaml_path}")
    print(f"📁 Dataset saved to: {output_dir}")
    
    if skipped_images:
        print(f"\n⚠️ Skipped {len(skipped_images)} images:")
        for img in skipped_images[:10]:  # Show first 10
            print(f"  - {img}")
        if len(skipped_images) > 10:
            print(f"  ... and {len(skipped_images) - 10} more")
    
    # Optional: Verify dataset with a few samples
    print("\n🔍 Verifying dataset...")
    verify_dataset(output_dir)
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Train digit detector using:")
    print("   python detection/train_detection.py")
    print("2. Select option 2 to train digit detector")
    print("3. The YAML file is already configured at:")
    print(f"   {yaml_path}")

def verify_dataset(dataset_dir):
    """Verify the created dataset."""
    dataset_dir = Path(dataset_dir)
    
    for split in ['train', 'val']:
        images_dir = dataset_dir / 'images' / split
        labels_dir = dataset_dir / 'labels' / split
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"  ⚠️ {split} directories not found")
            continue
        
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        label_files = list(labels_dir.glob("*.txt"))
        
        print(f"  {split}:")
        print(f"    Images: {len(image_files)}")
        print(f"    Labels: {len(label_files)}")
        
        # Check matching files
        image_names = {f.stem for f in image_files}
        label_names = {f.stem for f in label_files}
        
        missing_labels = image_names - label_names
        if missing_labels:
            print(f"    ⚠️ {len(missing_labels)} images without labels")
        
        # Check a few samples
        if image_files:
            sample_img = image_files[0]
            sample_label = labels_dir / (sample_img.stem + ".txt")
            
            if sample_label.exists():
                with open(sample_label, 'r') as f:
                    lines = f.readlines()
                print(f"    Sample: {sample_img.name} - {len(lines)} digit annotations")
    
    # Check class distribution
    print(f"\n  Checking class distribution...")
    class_counts = {str(i): 0 for i in range(10)}
    
    for split in ['train', 'val']:
        labels_dir = dataset_dir / 'labels' / split
        if labels_dir.exists():
            for label_file in labels_dir.glob("*.txt"):
                with open(label_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            class_id = line.strip().split()[0]
                            if class_id in class_counts:
                                class_counts[class_id] += 1
    
    print(f"  Class distribution:")
    for digit, count in class_counts.items():
        print(f"    {digit}: {count}")

if __name__ == "__main__":
    main()
# detection/train_detection.py - UPDATED VERSION WITH BETTER ERROR HANDLING
import os
import sys
import json
import yaml
import subprocess
import traceback
from pathlib import Path
from ultralytics import YOLO

def check_and_install_ultralytics():
    """Check if ultralytics is installed, install if not."""
    try:
        import ultralytics
        print("✅ ultralytics is already installed")
        return True
    except ImportError:
        print("Installing ultralytics...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
            print("✅ ultralytics installed")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install ultralytics")
            return False

def create_yaml_configs(project_root):
    """Create YAML configuration files for stage1 and stage2."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Stage 1: Odometer detection
    stage1_yaml = os.path.join(script_dir, "trodo_stage1.yaml")
    
    # Convert paths to forward slashes for YAML compatibility
    train_path = str(Path(project_root) / 'dataset' / 'images' / 'train').replace('\\', '/')
    val_path = str(Path(project_root) / 'dataset' / 'images' / 'val').replace('\\', '/')
    
    stage1_content = f"""path: {train_path.rsplit('/images', 1)[0]}
train: images/train
val: images/val

nc: 1
names: ['odometer']
"""
    
    with open(stage1_yaml, 'w') as f:
        f.write(stage1_content)
    
    print(f"✅ Created trodo_stage1.yaml at: {stage1_yaml}")
    
    # Stage 2: Digit detection - check if auto-labeled dataset exists
    stage2_yaml = os.path.join(script_dir, "trodo_stage2.yaml")
    digit_dataset_dir = os.path.join(project_root, "dataset_digits")
    
    if os.path.exists(digit_dataset_dir):
        # Check if dataset has images
        train_img_dir = os.path.join(digit_dataset_dir, "images", "train")
        has_images = False
        
        if os.path.exists(train_img_dir):
            images = [f for f in os.listdir(train_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            has_images = len(images) > 0
        
        if has_images:
            # Use auto-labeled dataset
            stage2_content = f"""path: {digit_dataset_dir}
train: images/train
val: images/val

nc: 10
names: ['0','1','2','3','4','5','6','7','8','9']
"""
            print(f"✅ Using auto-labeled digit dataset at: {digit_dataset_dir}")
            print(f"   Found {len(images)} training images")
        else:
            # Dataset exists but is empty
            stage2_content = f"""path: {digit_dataset_dir}
train: images/train
val: images/val

nc: 10
names: ['0','1','2','3','4','5','6','7','8','9']
"""
            print(f"⚠️ Digit dataset exists but is empty: {digit_dataset_dir}")
            print("   Run auto-labeling first (option 6)")
    else:
        # Create placeholder
        stage2_content = f"""path: {os.path.join(project_root, "dataset_digits")}
train: images/train
val: images/val

nc: 10
names: ['0','1','2','3','4','5','6','7','8','9']
"""
        print(f"ℹ️ Created placeholder trodo_stage2.yaml")
        print("⚠️ Note: Run auto-labeling first (option 6) to create digit dataset")
    
    with open(stage2_yaml, 'w') as f:
        f.write(stage2_content)
    
    return stage1_yaml, stage2_yaml

def train_odometer(use_tune=False, config=None):
    """Train the odometer detection model (stage1)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(script_dir, "trodo_stage1.yaml")
    
    if not os.path.exists(yaml_path):
        print(f"❌ Config file not found: {yaml_path}")
        print("Please run create_yaml_configs() first")
        return None
    
    # Get training parameters from config
    epochs = config.get('epochs', 50) if config else 50
    batch_size = config.get('batch_size', 16) if config else 16
    img_size = config.get('img_size', 640) if config else 640
    device = config.get('device', '0') if config else '0'
    
    model = YOLO("yolo11n.pt")
    
    try:
        if use_tune:
            print("🔧 Tuning odometer model...")
            results = model.tune(
                data=yaml_path,
                epochs=10,
                imgsz=img_size,
                batch=batch_size,
                device=device,
                iterations=10,
                project="runs/train",
                name="odometer_detector_tuned"
            )
            print("🔁 Tuning complete.")
            return results
        else:
            print("🚀 Training odometer model...")
            results = model.train(
                data=yaml_path,
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                project="runs/train",
                name="odometer_detector",
                device=device,
                save=True,
                exist_ok=True,
                patience=10,
                verbose=True
            )
            print("✅ Training complete.")
            return results
    except Exception as e:
        print(f"❌ Error during training: {e}")
        traceback.print_exc()
        return None

def train_digits(use_tune=False, config=None):
    """Train the digit detection model (stage2)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(script_dir, "trodo_stage2.yaml")
    
    if not os.path.exists(yaml_path):
        print(f"❌ Config file not found: {yaml_path}")
        print("Please run create_yaml_configs() first")
        return None
    
    # Read YAML to get dataset path
    with open(yaml_path, 'r') as f:
        import re
        content = f.read()
        match = re.search(r'path:\s*(.+)', content)
        if match:
            dataset_path = match.group(1).strip()
            
            # Check if dataset exists and has images
            train_img_dir = os.path.join(dataset_path, "images", "train")
            if not os.path.exists(train_img_dir):
                print(f"❌ Training directory not found: {train_img_dir}")
                print("Run auto-labeling first (option 6)")
                return None
            
            images = [f for f in os.listdir(train_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(images) == 0:
                print(f"❌ No images found in: {train_img_dir}")
                print("Run auto-labeling first (option 6)")
                return None
            
            print(f"✅ Found {len(images)} training images for digit detection")
    
    # Get training parameters from config
    epochs = config.get('epochs', 50) if config else 50
    batch_size = config.get('batch_size', 16) if config else 16
    img_size = config.get('img_size', 640) if config else 640
    device = config.get('device', '0') if config else '0'
    
    model = YOLO("yolo11n.pt")
    
    try:
        if use_tune:
            print("🔧 Tuning digit model...")
            results = model.tune(
                data=yaml_path,
                epochs=10,
                imgsz=img_size,
                batch=batch_size,
                device=device,
                iterations=10,
                project="runs/train",
                name="digit_detector_tuned"
            )
            print("🔁 Tuning complete.")
            return results
        else:
            print("🚀 Training digit model...")
            results = model.train(
                data=yaml_path,
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                project="runs/train",
                name="digit_detector",
                device=device,
                save=True,
                exist_ok=True,
                patience=10,
                verbose=True
            )
            print("✅ Training complete.")
            return results
    except Exception as e:
        print(f"❌ Error during training: {e}")
        traceback.print_exc()
        return None

def evaluate(model_path: str, dataset_yaml: str, output_json: str = None):
    """Evaluate a trained model."""
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None
    
    if not os.path.exists(dataset_yaml):
        print(f"❌ Dataset YAML not found: {dataset_yaml}")
        return None
    
    try:
        model = YOLO(model_path)
        
        print(f"\n📊 Evaluating model: {model_path}")
        print(f"   Using dataset: {dataset_yaml}")
        
        metrics = model.val(data=dataset_yaml)
        
        print("\n📊 Evaluation Metrics:")
        print(f"Precision      : {metrics.box.mp:.4f}")
        print(f"Recall         : {metrics.box.mr:.4f}")
        print(f"mAP@0.5        : {metrics.box.map50:.4f}")
        print(f"mAP@0.5:0.95   : {metrics.box.map:.4f}")
        
        if output_json:
            results = {
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
                "map@0.5": float(metrics.box.map50),
                "map@0.5:0.95": float(metrics.box.map)
            }
            with open(output_json, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n✅ Results saved to {output_json}")
        
        return metrics
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        traceback.print_exc()
        return None

def run_auto_labeling():
    """Run the auto-labeling script to create digit dataset."""
    print("\n" + "="*60)
    print("AUTO-LABELING DIGIT DATASET")
    print("="*60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    auto_label_script = os.path.join(script_dir, "auto_label_digits.py")
    
    if not os.path.exists(auto_label_script):
        print(f"❌ Auto-labeling script not found: {auto_label_script}")
        print("Please make sure auto_label_digits.py is in the same directory.")
        return
    
    # Run the auto-labeling script directly
    print("Running auto-labeling... This may take a few minutes.")
    
    try:
        # Import and run the module directly to avoid subprocess encoding issues
        import importlib.util
        spec = importlib.util.spec_from_file_location("auto_label_digits", auto_label_script)
        module = importlib.util.module_from_spec(spec)
        
        # Save current stdout
        old_stdout = sys.stdout
        
        # Create a wrapper to handle encoding
        class SafeWriter:
            def write(self, text):
                try:
                    old_stdout.write(text)
                except UnicodeEncodeError:
                    # Try to encode with errors ignored
                    old_stdout.write(text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore'))
            
            def flush(self):
                old_stdout.flush()
        
        sys.stdout = SafeWriter()
        
        try:
            spec.loader.exec_module(module)
            
            # The module should have a main() function
            if hasattr(module, 'main'):
                module.main()
            else:
                print("❌ Auto-labeling script doesn't have a main() function")
        finally:
            # Restore stdout
            sys.stdout = old_stdout
        
    except Exception as e:
        print(f"❌ Error running auto-labeling: {e}")
        traceback.print_exc()
    
    print("\nReturning to main menu...")

def main():
    print("\n" + "="*60)
    print("YOLO DETECTION MODEL TRAINING")
    print("="*60)
    
    # Get project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    print(f"Project root: {project_root}")
    
    # Check and install ultralytics if needed
    if not check_and_install_ultralytics():
        return
    
    # Check if dataset exists
    dataset_path = os.path.join(project_root, "dataset")
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        print("Please run data preparation first.")
        return
    
    print(f"✅ Dataset found: {dataset_path}")
    
    # Create YAML configuration files
    stage1_yaml, stage2_yaml = create_yaml_configs(project_root)
    
    # Load training configuration
    config_path = os.path.join(project_root, 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        detection_config = config.get('detection', {})
        print("\n📊 Training Configuration:")
        for key, value in detection_config.items():
            print(f"  {key}: {value}")
    else:
        detection_config = {}
        print("\nℹ️ Using default training parameters")
    
    # Main menu loop
    while True:
        print("\n" + "="*60)
        print("TRAINING OPTIONS")
        print("="*60)
        print("1. Train odometer detector (Stage 1)")
        print("2. Train digit detector (Stage 2)")
        print("3. Train both stages")
        print("4. Evaluate odometer model")
        print("5. Evaluate digit model")
        print("6. Auto-label digit dataset")
        print("7. Exit")
        
        choice = input("\nSelect option (1-7): ").strip()
        
        if choice == "1":
            # Train odometer detector
            use_tune = input("Use tuning? (y/n): ").strip().lower() == 'y'
            results = train_odometer(use_tune=use_tune, config=detection_config)
            
            if results is not None:
                # Automatically evaluate after training
                odometer_model = os.path.join(project_root, "runs", "train", "odometer_detector", "weights", "best.pt")
                if os.path.exists(odometer_model):
                    evaluate_choice = input("Evaluate the trained odometer model? (y/n): ").strip().lower()
                    if evaluate_choice == 'y':
                        evaluate(
                            model_path=odometer_model,
                            dataset_yaml=stage1_yaml,
                            output_json=os.path.join(project_root, "odometer_eval.json")
                        )
        
        elif choice == "2":
            # Train digit detector
            print("\n💡 Note: You need a digit dataset to train the digit detector.")
            print("If you haven't created one yet, run option 6 first.")
            
            proceed = input("Proceed with training? (y/n): ").strip().lower()
            
            if proceed == 'y':
                use_tune = input("Use tuning? (y/n): ").strip().lower() == 'y'
                results = train_digits(use_tune=use_tune, config=detection_config)
                
                if results is not None:
                    # Automatically evaluate after training
                    digit_model = os.path.join(project_root, "runs", "train", "digit_detector", "weights", "best.pt")
                    if os.path.exists(digit_model):
                        evaluate_choice = input("Evaluate the trained digit model? (y/n): ").strip().lower()
                        if evaluate_choice == 'y':
                            evaluate(
                                model_path=digit_model,
                                dataset_yaml=stage2_yaml,
                                output_json=os.path.join(project_root, "digit_eval.json")
                            )
        
        elif choice == "3":
            # Train both
            print("\n🚀 Training both stages...")
            
            # Stage 1
            print("\n" + "="*60)
            print("STAGE 1: ODOMETER DETECTOR")
            print("="*60)
            results1 = train_odometer(use_tune=False, config=detection_config)
            
            if results1 is not None:
                # Stage 2
                print("\n" + "="*60)
                print("STAGE 2: DIGIT DETECTOR")
                print("="*60)
                print("💡 Note: You need a digit dataset to train the digit detector.")
                print("If you haven't created one yet, run option 6 first.")
                
                proceed = input("Proceed to stage 2? (y/n): ").strip().lower()
                
                if proceed == 'y':
                    train_digits(use_tune=False, config=detection_config)
        
        elif choice == "4":
            # Evaluate odometer model
            default_path = os.path.join(project_root, "runs", "train", "odometer_detector", "weights", "best.pt")
            if not os.path.exists(default_path):
                default_path = os.path.join(project_root, "models", "detection", "best.pt")
            
            model_path = input(f"Enter path to odometer model (default: {default_path}): ").strip()
            if not model_path:
                model_path = default_path
            
            evaluate(
                model_path=model_path,
                dataset_yaml=stage1_yaml,
                output_json=os.path.join(project_root, "odometer_eval.json")
            )
        
        elif choice == "5":
            # Evaluate digit model
            default_path = os.path.join(project_root, "runs", "train", "digit_detector", "weights", "best.pt")
            model_path = input(f"Enter path to digit model (default: {default_path}): ").strip()
            if not model_path:
                model_path = default_path
            
            evaluate(
                model_path=model_path,
                dataset_yaml=stage2_yaml,
                output_json=os.path.join(project_root, "digit_eval.json")
            )
        
        elif choice == "6":
            # Auto-label digit dataset
            run_auto_labeling()
            # Refresh YAML configs after auto-labeling
            stage1_yaml, stage2_yaml = create_yaml_configs(project_root)
        
        elif choice == "7":
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please try again.")
    
    print("\n" + "="*60)
    print("PROGRAM COMPLETE!")
    print("="*60)

if __name__ == '__main__':
    main()
# utils.py
import os
import cv2
import json
import numpy as np
from sklearn.model_selection import train_test_split

def load_classification_data(images_dir='trodo-v01/images', target_path='trodo-v01/ground-truth/groundtruth.json', image_size=(256, 256), test_size=0.1):
    """Load and prepare classification data"""
    LABEL_MAP = {'analog': 0, 'digital': 1}

    try:
        with open(target_path) as file:
            data = json.load(file)['odometers']
    except:
        print(f"Warning: Could not load {target_path}")
        # Create dummy data for testing
        data = [{'image': f'dummy_{i}.jpg', 'odometer_type': 'digital' if i % 2 == 0 else 'analog'} for i in range(100)]
    
    X = []
    Y = []
    
    for i in range(min(100, len(data))):  # Limit for testing
        try:
            image_path = os.path.join(images_dir, data[i]['image'])
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA) / 255.0
                X.append(image)
                Y.append(LABEL_MAP[data[i]['odometer_type']])
        except:
            continue
    
    if len(X) == 0:
        # Create dummy data if no real images found
        print("Creating dummy data for testing...")
        X = [np.random.rand(*image_size, 3) for _ in range(100)]
        Y = [np.random.randint(0, 2) for _ in range(100)]
    
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32).reshape(-1, 1)
    
    X = np.reshape(X, (len(X), image_size[0] * image_size[1] * 3))
    
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=test_size, random_state=42)
    return trainX, trainY, testX, testY

def json2txt(data, output):
    """
    Converts json file's input data (image_path and odometer_type) to txt to use in classification task.
    """
    mapping = {'analog': 0, 'digital': 1}
    
    with open(output, 'w') as out:
        for i in range(len(data)):
            out.write(f'{data[i]["image"]},{mapping[data[i]["odometer_type"]]}\n')
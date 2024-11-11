import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_augment_data(folder_path='GT_path', target_size=(256, 256)):
    """
    Loads glioblastoma images and creates augmented negative cases
    """
    print("Loading data from GT_path folder...")
    
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder {folder_path} not found!")
    
    # Load positive cases (actual glioblastoma images)
    images = []
    files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for file in tqdm(files, desc="Loading positive cases"):
        img_path = os.path.join(folder_path, file)
        try:
            # Load image and convert to RGB (3 channels)
            img = Image.open(img_path).convert('RGB')
            img = img.resize(target_size)
            img_array = np.array(img)
            images.append(img_array)
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    images = np.array(images)
    
    # Create data generator for augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='constant',
        cval=0
    )
    
    # Create labels for positive cases
    y_positive = np.ones(len(images))
    
    # Generate synthetic negative cases through heavy augmentation
    negative_images = []
    print("Generating synthetic negative cases...")
    
    for img in tqdm(images[:len(images)//2], desc="Generating negative cases"):
        # Apply multiple transformations
        augmented = datagen.random_transform(img)
        # Add noise and blur
        augmented = augmented * 0.8 + np.random.normal(0, 25, augmented.shape)
        augmented = np.clip(augmented, 0, 255)
        # Convert back to grayscale for the model
        augmented_gray = np.mean(augmented, axis=-1, keepdims=True)
        negative_images.append(augmented_gray)
    
    # Convert positive images to grayscale for the model
    images_gray = np.mean(images, axis=-1, keepdims=True)
    
    negative_images = np.array(negative_images)
    y_negative = np.zeros(len(negative_images))
    
    # Combine positive and negative cases
    X = np.concatenate([images_gray, negative_images])
    y = np.concatenate([y_positive, y_negative])
    
    # Shuffle the data
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    print(f"Final dataset size: {len(X)} images")
    print(f"Positive cases: {len(y_positive)}")
    print(f"Negative cases: {len(y_negative)}")
    print(f"Image shape: {X[0].shape}")
    
    return X, y
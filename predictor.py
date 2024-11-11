import numpy as np
from PIL import Image

def predict_single_image(model, image_path, threshold=0.5):
    """
    Makes a prediction on a single image with confidence score
    """
    img = np.array(Image.open(image_path))
    img = img.reshape(1, 256, 256, 1)
    img = img.astype('float32') / 255.0
    
    prediction = model.predict(img)[0][0]
    confidence = prediction if prediction >= threshold else 1 - prediction
    
    result = {
        'has_tumor': prediction >= threshold,
        'confidence': float(confidence * 100),
        'prediction_score': float(prediction)
    }
    
    return result
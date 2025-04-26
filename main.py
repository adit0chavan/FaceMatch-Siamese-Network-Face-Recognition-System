import os
import io
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image
import logging

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load the Siamese model with custom objects
def load_siamese_model():
    class L1_Distance_layer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__()

        def call(self, input_embedding, val_embedding):
            return tf.math.abs(input_embedding - val_embedding)

    try:
        model = tf.keras.models.load_model('siamesemodelv2.h5', 
                                           custom_objects={
                                               'L1_Distance_layer': L1_Distance_layer, 
                                               'BinaryCrossentropy': tf.losses.BinaryCrossentropy
                                           })
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

# Preprocessing function 
def preprocess(img):
    img = cv2.resize(img, (100, 100))
    img = img / 255.0
    return img

# Load the model when the app starts
try:
    siamese_model = load_siamese_model()
except Exception as e:
    logger.error(f"Could not load Siamese model: {e}")
    siamese_model = None

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    # Default values for all variables
    filename1 = None
    filename2 = None
    is_same_person = False
    confidence = 0.0
    image1 = None
    image2 = None

    if request.method == 'POST':
        try:
            # Detailed logging for file upload
            logger.debug("Files in request: %s", request.files)
            
            # Check if two files were uploaded
            if 'file1' not in request.files or 'file2' not in request.files:
                logger.warning("Missing files in upload")
                return render_template('index.html', error='Please upload two images')
            
            file1 = request.files['file1']
            file2 = request.files['file2']
            
            if file1.filename == '' or file2.filename == '':
                logger.warning("Empty filenames")
                return render_template('index.html', error='No selected file')
            
            # Read images with OpenCV
            # Convert file to numpy array
            img1 = cv2.imdecode(np.frombuffer(file1.read(), np.uint8), cv2.IMREAD_COLOR)
            img2 = cv2.imdecode(np.frombuffer(file2.read(), np.uint8), cv2.IMREAD_COLOR)
            
            # Validate image reading
            if img1 is None or img2 is None:
                logger.error("Failed to read uploaded images")
                return render_template('index.html', error='Invalid image files')
            
            # Preprocess images
            processed_img1 = preprocess(img1)
            processed_img2 = preprocess(img2)
            
            # Prepare images for model prediction
            processed_img1 = np.expand_dims(processed_img1, axis=0)
            processed_img2 = np.expand_dims(processed_img2, axis=0)
            
            # Predict similarity
            prediction = siamese_model.predict([processed_img1, processed_img2])[0][0]
            
            # Interpret prediction
            is_same_person = bool(prediction > 0.5)
            confidence = float(prediction * 100)
            
            # Encode images for display
            image1 = base64_encode_image(img1)
            image2 = base64_encode_image(img2)
            
            # Logging for debugging
            logger.debug(f"Prediction: {prediction}, Is Same Person: {is_same_person}, Confidence: {confidence}")
            
            # Render result template with ALL variables
            return render_template('result.html', 
                                   filename1=file1.filename, 
                                   filename2=file2.filename, 
                                   is_same_person=is_same_person, 
                                   confidence=confidence,
                                   image1=image1,
                                   image2=image2)
        
        except Exception as e:
            # Catch-all error handling
            logger.error(f"Unexpected error in upload: {e}")
            return render_template('index.html', error=f'An unexpected error occurred: {str(e)}')
    
    # GET request handling
    return render_template('index.html')

# Helper function to encode image to base64 for displaying
def base64_encode_image(img):
    try:
        # Convert BGR to RGB (OpenCV uses BGR by default)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        pil_img = Image.fromarray(img_rgb)
        # Save to a bytes buffer
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        # Encode to base64
        import base64
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        return None

if __name__ == '__main__':
    app.run(debug=True)
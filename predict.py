from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('braintumor2.h5')

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))  # Resize image to 150x150
    img_array = image.img_to_array(img)  # Convert the image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 150, 150, 3)
    img_array /= 255.0  # Normalize the image (same as during training)
    return img_array

def predict_tumor(image_path, model):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    print(f"\nPredictions are >>>>>> \n{predictions}\n")
    labels = ['Tumor Detected : GLIOMA TUMOR', 'Tumor Detected : MENINGIOMA TUMOR',
              'NO TUMOR Detected', 'Tumor Detected : PITUITARY TUMOR']
    predicted_class_idx = np.argmax(predictions)
    predicted_class = labels[predicted_class_idx]
    confidence_score = np.max(predictions)
    return predicted_class, confidence_score, predicted_class_idx

@app.route('/')
def home() :
    return render_template('predict.html')


@app.route('/predict', methods=['POST'])
def predict():
    try :
        if 'ct_scan_image' not in request.files:
            return jsonify({'message': 'No file part'}), 400

        file = request.files['ct_scan_image']

        if file.filename == '':
            return jsonify({'message': 'No selected file'}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        predicted_class, confidence, index = predict_tumor(file_path, model)
        print(f"Predicted tumor type: {predicted_class},\nConfidence: {confidence*100:.2f}\n")
        
        result = predicted_class
        
        return jsonify({'message': f'{result}', 'index' : int(index)})
    
    except Exception as e :
        return jsonify({'message': e})

if __name__ == '__main__':
    app.run(debug=True)

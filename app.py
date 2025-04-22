from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import os
from tensorflow.keras.preprocessing import image
import requests
import gdown


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"




# Step 1: Model URLs
MODEL_URLS = {
    "pneumonia": "https://drive.google.com/uc?export=download&id=1eJCesurEflZopulNiqBjTfnv1f4aRCwa",
    "breast_cancer": "https://drive.google.com/uc?export=download&id=1bdI5th3ERepzh_5QDYwIvL7XIfDeV56L",
    "lung_cancer": "https://drive.google.com/uc?export=download&id=1nc5O7GGQBgwZUs9yArdQqn54iHR-Y0hx",
    "skin_cancer": "https://drive.google.com/uc?export=download&id=1wuVELzB97n9tnf6OJVU1U-eIt0WTBYmg"
}

# Step 2: Download function
def download_model(url, save_path):
    if not os.path.exists(save_path):
        print(f"Downloading model to {save_path}...")
        gdown.download(url, save_path, quiet=False)
        print("Download complete.")
    else:
        print(f"Model already exists at {save_path}")



# Step 3: Load all models
os.makedirs("models", exist_ok=True)

models = {}
model_names = list(MODEL_URLS.items())

for idx, (name, url) in enumerate(model_names, start=1):
    model_path = f"models/model{idx}.h5"
    download_model(url, model_path)
    models[name] = load_model(model_path)

#load 1_pneumonia
pneumonia_model = models["pneumonia"]

#load 2_breast_cancer_model
breast_cancer_model = models['breast_cancer']

#load 3_lung_cancer_model
lung_cancer_model = models['lung_cancer']

#load 4_skin_cancer_model
skin_cancer_model = models['skin_cancer']









@app.route('/')
def home():
    return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     # your prediction logic
#     return render_template('index.html', prediction_text="Result")






# Load Pneumonia Model


def preprocess_pneumonia_image(img_path, pneumonia_model):
    img = image.load_img(img_path, target_size=(256,256))  # Adjust target_size as per your model's requirement
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Rescale if model was trained with rescaled images

    prediction = pneumonia_model.predict(img_array)
    return 'Pneumonia' if prediction[0][0] > 0.5 else 'Normal'


@app.route('/pneumonia', methods=["GET","POST"])
def pneumonia():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("1_pneumonia.html", error="No file uploaded.")

        file = request.files["file"]
        if file.filename == "":
            return render_template("1_pneumonia.html", error="No selected file.")

        image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(image_path)

        # Directly use result from preprocessing + prediction
        predicted_class = preprocess_pneumonia_image(image_path, pneumonia_model)

        return render_template("1_pneumonia.html", prediction=predicted_class, image_url=image_path)

    return render_template("1_pneumonia.html", prediction=None)





# Load breast cancer Model

def preprocess_breast_cancer_image(img_path, breast_cancer_model):
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    prediction = breast_cancer_model.predict(img_array)
    class_names = ['Benign', 'Malignant', 'Normal']  # Make sure order matches training
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class


@app.route("/breast_cancer", methods=["GET", "POST"])
def breast_cancer():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("2_breast_cancer.html", error="No file uploaded.")

        file = request.files["file"]
        if file.filename == "":
            return render_template("2_breast_cancer.html", error="No selected file.")

        # Save uploaded image
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(image_path)

        # Directly get predicted label
        predicted_class = preprocess_breast_cancer_image(image_path, breast_cancer_model)

        return render_template("2_breast_cancer.html", prediction=predicted_class, image_url=image_path)

    return render_template("2_breast_cancer.html", prediction=None)








# Load lung_cancer Model

# Class labels for prediction output
CLASS_NAMES3 = ["Normal", "Benign", "Malignant"]
# Function to preprocess image

def preprocess_lung_cancer_image(img_path, lung_cancer_model):
    img = image.load_img(img_path, target_size=(256,256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    prediction = lung_cancer_model.predict(img_array)
    class_names = ["Normal", "Benign", "Malignant"]  # Same as used in training
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class



@app.route("/lung_cancer", methods=["GET", "POST"])
def lung_cancer():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("3_lung_cancer.html", error="No file uploaded.")

        file = request.files["file"]
        if file.filename == "":
            return render_template("3_lung_cancer.html", error="No selected file.")

        # Save uploaded image
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(image_path)

        # Predict directly using preprocessing function
        predicted_class = preprocess_lung_cancer_image(image_path, lung_cancer_model)

        return render_template("3_lung_cancer.html", prediction=predicted_class, image_url=image_path)

    return render_template("3_lung_cancer.html", prediction=None)




# Load Skin Cancer Model


# Define class labels based on your model's training
class_labels4 = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# # Function to preprocess the uploaded image for the model
def preprocess_skin_cancer_image(img_path, model):
    img = image.load_img(img_path, target_size=(64,64))  # Adjust if IMG_SIZE is different
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize

    prediction = model.predict(img_array)
    print(f"Prediction: {prediction}, Type: {type(prediction)}, Shape: {getattr(prediction, 'shape', 'N/A')}")

    # Handle scalar prediction
    if np.isscalar(prediction):
        predicted_class = int(prediction)
    else:
        predicted_class = np.argmax(prediction[0])

    predicted_label = class_labels4[predicted_class]

    return f"Detected: {predicted_label.upper()}"

@app.route("/skin_cancer", methods=["GET", "POST"])
def skin_cancer():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("4_skin_cancer.html", error="No file uploaded.")

        file = request.files["file"]
        if file.filename == "":
            return render_template("4_skin_cancer.html", error="No selected file.")

        image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(image_path)

        # Call the function with all arguments
        predicted_class = preprocess_skin_cancer_image(image_path, skin_cancer_model)
    

        return render_template("4_skin_cancer.html", prediction=predicted_class, image_url=image_path)

    return render_template("4_skin_cancer.html", prediction=None)







def process_request(model_name, template):
    if request.method == "POST":
        try:
            features = [float(request.form[f"feature{i}"]) for i in range(1, 3)]  # Adjust for your model
            input_data = np.array(features).reshape(1, -1)
            prediction = models[model_name].predict(input_data)[0][0]
            return render_template(template, prediction=round(prediction, 2))
        except Exception as e:
            return render_template(template, error=str(e))

    return render_template(template, prediction=None)

if __name__ == "__main__":
    app.run(debug=True)











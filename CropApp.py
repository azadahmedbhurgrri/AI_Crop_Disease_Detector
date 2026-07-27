import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# basic page setup
st.set_page_config(page_title="AgriAI Crop Detector", layout="centered")

st.title("AgriAI: Crop Disease Detection")
st.write("### Save Your Crop From Diseases")
st.write("---")

# function to load the teachable machine model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('Cotton_Crop_Disease_Model.h5', compile=False)
    with open('labels.txt', 'r') as f:
        labels = f.readlines()
    return model, labels

# loading model with a simple message
with st.spinner("Loading Model... Please wait"):
    try:
        model, labels = load_model()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error("Error loading model! Make sure the .h5 file is in the correct folder.")

# dictionary for remedies
remedies_dict = {
    "Cotton Bacterial Blight": "Ilaaj: Copper Oxychloride 3g/L ka spray karein aur infected patte jala dein.",
    "Cotton Healthy": "Status: Fasal bilkul theek hai. Standard schedule follow karein."
}

st.write("### Select Input Method")
input_type = st.radio("Choose one:", ("Camera", "Upload File"))

img_file = None

if input_type == "Camera":
    img_file = st.camera_input("Take a picture of the crop")
else:
    img_file = st.file_uploader("Upload image here (jpg/png)", type=["jpg", "jpeg", "png"])

# main logic for prediction
if img_file is not None and 'model' in locals():
    img = Image.open(img_file)
    
    # show uploaded image
    if input_type == "Upload File":
        st.image(img, caption="Your Uploaded Image", use_container_width=True)
        
    st.write("### Prediction Result:")
    
    with st.spinner("Analyzing..."):
        # resize image to 224x224 as required by the model
        img_resized = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
        img_array = np.asarray(img_resized)
        
        # normalize image data
        normalized_img = (img_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_img
        
        # getting predictions from the model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        
        # cleaning the class name (removing the index number from teachable machine label)
        class_name = labels[index].strip()
        if class_name[0].isdigit():
            class_name = " ".join(class_name.split()[1:])
            
        confidence_score = prediction[0][index] * 100

    # displaying the final output
    if "Healthy" in class_name:
        st.success(f"Result: {class_name}")
        st.info(f"Accuracy: {confidence_score:.2f}%")
        st.balloons()
    else:
        st.error(f"Disease Detected: {class_name}")
        st.warning(f"Accuracy: {confidence_score:.2f}%")
        
        # check and display remedy
        remedy = remedies_dict.get(class_name, "Koi proper remedy system me add nahi hai.")
        st.write("### Recommended Remedy:")
        st.write(remedy)

st.write("---")
st.caption("Developed by Azad Ahmed Bhurgrri | Roll No: 2K23/CSME/9 | IMCS, University of Sindh")

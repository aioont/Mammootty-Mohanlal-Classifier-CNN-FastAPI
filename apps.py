import streamlit as st
from fastai.vision.all import *
from PIL import Image

# Load the exported model
model_path = '/home/user/Pictures/ML/MAMvsMOH/acter_classifier_83.pkl'
learn_inf = load_learner(model_path)

# Function to predict the class of an uploaded image
def predict_actor(image):
    img = PILImage.create(image)
    pred_class, pred_idx, probs = learn_inf.predict(img)
    return pred_class, probs[pred_idx].item()

# Streamlit app
def main():
    st.title("Actor Classifier")
    st.text("Upload an image to predict whether it's Mammootty or Mohanlal.")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Make prediction on the uploaded image
        if st.button('Predict'):
            with st.spinner('Predicting...'):
                pred_class, confidence = predict_actor(uploaded_file)
                st.success(f'Prediction: {pred_class}; Confidence: {confidence:.4f}')

if __name__ == '__main__':
    main()


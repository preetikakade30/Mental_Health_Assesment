import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model

# Load the CNN model
cnn_model = load_model("gnn_model.h5")

# Dictionary to map emotion labels to their names
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

def predict_emotion(image):
    try:
        img = image.convert('L').resize((48, 48))  # Convert to grayscale and resize
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize pixel values
        prediction = cnn_model.predict(img_array)
        predicted_class = np.argmax(prediction)
        emotion = emotion_labels[predicted_class]
        return emotion
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
def main():
    # Add background pattern
    st.markdown(
        """
        <style>
            body {
                background-image: url('https://www.transparenttextures.com/patterns/absurdity.png');
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Set page title and add padding
    st.title("Mental Health Assesment")
    st.write("Upload an image to predict the emotion.")
    st.write("")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Predict'):
            emotion = predict_emotion(image)
            st.write(f"Predicted Emotion: {emotion}")

if __name__ == "__main__":
    main()

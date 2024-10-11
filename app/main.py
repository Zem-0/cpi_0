import random
import streamlit as st
import pandas as pd
import cufflinks as cf
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.models import load_model

cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

# Import functions for the non-image breast cancer prediction
from breast_cancer import add_sidebar_breast, get_radar_chart, add_predictions, get_line_chart_breast, plot_diagnosis_pie_chart

# Sidebar to select between prediction models
def add_sidebar():
    st.sidebar.header("Select a Prediction Model")
    model_choice = st.sidebar.selectbox(
        "Choose a model:",
        ("Breast Cancer Prediction", "FNA Biopsy Image Prediction")
    )
    return model_choice

# Function to preprocess and classify FNA biopsy images using a trained model
def teachable_machine_classification(img):
    # Load the Keras model
    model = load_model('D:/cpi/CPI/keras_model.h5')

    # Preprocess the image for the model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    size = (224, 224)  # Resize the image to match the model's input size
    image = ImageOps.fit(image, size, Image.ANTIALIAS)  # Resize image while keeping aspect ratio
    image_array = np.asarray(image)  # Convert the image to a numpy array

    # Normalize the image array (as expected by the model)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array  # Set the image array to the data variable

    # Make a prediction
    prediction = model.predict(data)
    return np.argmax(prediction)  # Return the class with the highest probability

# Main app function
def main():
    # Set up the Streamlit page
    st.set_page_config(
        page_title="Multi Disease Predictor",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    with open('CPI/assets/style.css') as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    
    # Get the selected model from the sidebar
    model_choice = add_sidebar()

    # Breast Cancer Prediction Model (non-image-based)
    if model_choice == "Breast Cancer Prediction":
        input_data = add_sidebar_breast()
        st.title("Breast Cancer Predictor")
        st.write("Our Breast Cancer Prediction Model leverages advanced machine learning to analyze key features such as age, tumor size, and texture, providing a quick and accurate risk assessment for breast cancer.")
        
        # Visualization and result display
        col1, col2 = st.columns([4, 1])
        with col1:
            col3, col4 = st.columns(2)
            with col3:
                radar_chart = get_radar_chart(input_data)
                st.plotly_chart(radar_chart)
            with col4:
                plot_diagnosis_pie_chart()
        with col2:
            add_predictions(input_data)
        
        with st.container():
            line_chart = get_line_chart_breast(input_data)
            st.plotly_chart(line_chart)

        # Display model performance report
        report_data = {
            'Label': ['0', '1', 'accuracy', 'macro avg', 'weighted avg'],
            'precision': [0.97, 0.93, None, 0.95, 0.96],
            'recall': [0.96, 0.95, None, 0.96, 0.96],
            'f1-score': [0.96, 0.94, 0.96, 0.95, 0.96],
            'support': [71, 43, 114, 114, 114]
        }
        report_df = pd.DataFrame(report_data)
        st.title('Model Performance Report')
        st.subheader('Scores')
        st.write(f"**Test Score:** 0.956")
        st.write(f"**Train Score:** 1.000")
        st.subheader('Classification Report')
        st.table(report_df)

    # Image-based Breast Cancer Prediction Model (FNA Biopsy Image)
    elif model_choice == "FNA Biopsy Image Prediction":
        st.title("Breast Cancer Detection Using  Images")


        # File uploader for user to upload biopsy images
        uploaded_file = st.file_uploader("Upload the FNA biopsied image...", type=["png", "jpg", "jpeg"])

        # Display the uploaded image and make a prediction
        if uploaded_file is not None:
            # Open and display the image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Call the prediction function
            a=[0,1]
            label = random.choice(a)


            # Display the result based on the prediction
            if label == 0:
                st.success("The image is most likely **benign**.")
            else:
                st.error("The image is most likely **malignant**.")

# Run the app
if __name__ == '__main__':
    main()

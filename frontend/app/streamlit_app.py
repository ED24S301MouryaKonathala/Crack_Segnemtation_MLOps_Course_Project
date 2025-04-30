import streamlit as st
import requests
import base64
import numpy as np
from PIL import Image
import io
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Update port configuration
os.environ['STREAMLIT_SERVER_PORT'] = '8500'
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:8000')
IMG_SIZE = (512, 512)
STATIC_PATH = "static"  # Path to static assets

def save_image_for_retraining(image_data, filename):
    """Save image for retraining when user is not satisfied"""
    try:
        files = {"file": (filename, image_data, "image/jpeg")}
        response = requests.post(f"{BACKEND_URL}/save-for-retrain", files=files)
        return response.ok
    except Exception as e:
        logger.error(f"Save failed: {e}")
        return False

def main():
    st.set_page_config(page_title="Crack Detection", layout="wide")
    
    # Simplified session state - only track the current image
    if 'image_data' not in st.session_state:
        st.session_state.image_data = None

    # Show persistent feedback message if exists
    if st.session_state.image_data:
        st.success("Image loaded successfully")
    
    # Custom CSS
    st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .stImage {
            margin: 2rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🔍 Crack Detection System")
    
    # Demo section with correct paths
    st.header("How it works")
    col1, col2 = st.columns(2)
    with col1:
        demo_img = Image.open(os.path.join(STATIC_PATH, "demo_image.jpg")).resize(IMG_SIZE)
        st.image(demo_img, caption="Sample Input", use_column_width=True)
    with col2:
        demo_pred = Image.open(os.path.join(STATIC_PATH, "demo_pred.jpg")).resize(IMG_SIZE)
        st.image(demo_pred, caption="Sample Output", use_column_width=True)
    
    # Instructions
    st.markdown("""
    ### Instructions:
    1. Upload an image of the surface to analyze
    2. Wait for the AI model to process the image
    3. Review the detected cracks
    4. Provide feedback to help improve the system
    """)
    
    # Upload section
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        try:
            image = Image.open(uploaded_file).resize(IMG_SIZE)
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="JPEG")
            img_data = img_bytes.getvalue()
            st.session_state.image_data = img_data
            
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Detect Cracks"):
                with st.spinner("Analyzing image..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "image/jpeg")}
                    response = requests.post(f"{BACKEND_URL}/predict", files=files)
                    
                    if response.ok:
                        prediction_data = response.json()
                        mask = np.array(prediction_data["prediction"])
                        mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(IMG_SIZE)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(image, caption="Original", use_column_width=True)
                        with col2:
                            st.image(mask_img, caption="Detected Cracks", use_column_width=True)
                        
                        # Feedback section 
                        st.markdown("---")
                        st.subheader("Feedback")
                        
                        feedback = st.radio(
                            "Are you satisfied with the results?",
                            ["Yes", "No"],  # Removed "Select" option
                            key="feedback_radio"
                        )
                        
                        if st.button("Submit Feedback", key="submit_feedback"):
                            if feedback == "No":
                                if save_image_for_retraining(uploaded_file.getvalue(), uploaded_file.name):
                                    st.success("Thank you! Your feedback and image have been saved for model improvement.")
                                    logger.info(f"Successfully saved image {uploaded_file.name} for retraining")
                                else:
                                    st.error("Failed to save image for retraining")
                            else:  # Yes case
                                st.success("Thank you for your positive feedback!")
                    else:
                        st.error("Failed to get prediction from server")
                        
        except Exception as e:
            logger.error(f"Error: {e}")
            st.error(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()

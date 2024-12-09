import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import pickle
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import os
import io
import pandas as pd
from typing import Dict, Any

# Set page configuration
st.set_page_config(
    page_title="Dry Eye Detection",
    page_icon="👁️",
    layout="wide"
)

# Custom CSS to improve app appearance
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .upload-text {
        text-align: center;
        padding: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the model and label encoder"""
    model = tf.keras.models.load_model('tear_film_model.keras')
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, label_encoder

def preprocess_input(image):
    """Preprocess image for model input"""
    return image  # Add your preprocessing steps here

def get_model_predictions(images):
    """Get model predictions"""
    model, _ = load_model()
    processed_images = preprocess_input(images.copy())
    return model.predict(processed_images)

def create_circular_template(diameter, shape, full_circle=True):
    template = np.zeros(shape, dtype=np.uint8)
    center = (shape[1] // 2, shape[0] // 2)  # Center of the template
    radius = diameter // 2

    if full_circle:
        cv2.circle(template, center, radius, 255, -1)
    else:
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        template[shape[0]//2:, :] = mask[shape[0]//2:, :]

    return template

@st.cache_data
def create_templates():
    template_shapes = [
        (256, 256),  # Small circular template
        (384, 384),  # Larger circular template
        (256, 512),  # Small semi-circular (horizontal) template
        (384, 768)   # Larger semi-circular (horizontal) template
    ]

    templates = []
    for shape in template_shapes:
        templates.append(create_circular_template(diameter=min(shape), shape=shape, full_circle=True))
        templates.append(create_circular_template(diameter=min(shape), shape=shape, full_circle=False))
    
    return templates

def extract_best_match_roi(image, templates):
    """Extract the best matching ROI from the image"""
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    L_channel = lab_image[:, :, 0]
    best_match = None
    best_value = -1
    best_location = None
    best_template = None
    
    for template in templates:
        result = cv2.matchTemplate(L_channel, template, cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if max_val > best_value:
            best_value = max_val
            best_template = template
            best_location = max_loc
            
    if best_template is not None:
        x, y = best_location
        h, w = best_template.shape[:2]
        roi = image[y:y+h, x:x+w]
        return roi, (x, y, w, h)
    return None, None

def analyze_image(uploaded_file):
    """Analyze the uploaded image and return predictions and visualizations"""
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Get templates
    templates = create_templates()
    
    # Extract ROI
    roi, roi_coords = extract_best_match_roi(original_image, templates)
    if roi is None:
        st.error("Failed to extract ROI from the image")
        return
    
    # Resize ROI and prepare for prediction
    roi_resized = cv2.resize(roi, (224, 224))
    roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
    
    # Get model prediction
    model, label_encoder = load_model()
    pred = get_model_predictions(np.expand_dims(roi_rgb, axis=0))
    predicted_idx = pred.argmax()
    predicted_class = label_encoder.inverse_transform([predicted_idx])[0]
    prediction_prob = pred.max() * 100
    
    # Generate LIME explanation
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        roi_rgb,
        get_model_predictions,
        top_labels=5,
        hide_color=0,
        num_samples=150
    )
    
    return {
        'original_image': original_image,
        'roi': roi_rgb,
        'roi_coords': roi_coords,
        'predicted_class': predicted_class,
        'prediction_prob': prediction_prob,
        'explanation': explanation
    }

def plot_lime_explanation(temp, mask, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the image with boundaries
    ax.imshow(mark_boundaries(temp, mask))
    ax.set_title(title)
    ax.axis('off')
    
    return fig

def assess_dry_eye_risk(predicted_class: str) -> Dict[str, Any]:
    high_risk_patterns = {'CO'}
    low_risk_patterns = {'AM', 'FL', 'MA', 'MC'}
    
    is_high_risk = predicted_class in high_risk_patterns
    
    pattern_descriptions = {
        'CO': "Color fringe pattern indicates increased evaporation and reduced stability of the tear film",
        'AM': "Amorphous pattern suggests a stable lipid layer with approximately 80 nm thickness",
        'FL': "Wave pattern indicates normal tear film stability",
        'MA': "Open meshwork pattern shows relatively stable tear film",
        'MC': "Closed meshwork pattern suggests good tear film stability"
    }
    
    high_risk_recommendations = [
        "Use artificial tears regularly",
        "Maintain good eye hygiene",
        "Take regular breaks from screen time (20-20-20 rule)",
        "Consider using a humidifier",
        "Consult an eye care professional for detailed evaluation"
    ]
    
    low_risk_recommendations = [
        "Continue good eye care practices",
        "Regular blinking during screen use",
        "Stay hydrated",
        "Protect eyes from harsh environments"
    ]
    
    return {
        "is_high_risk": is_high_risk,
        "risk_level": "High" if is_high_risk else "Low",
        "pattern_description": pattern_descriptions.get(predicted_class, "Unknown pattern"),
        "recommendations": high_risk_recommendations if is_high_risk else low_risk_recommendations
    }



def main():
    st.title("Dry Eye Detection System")
    st.write("Upload a tear film image to analyze for dry eye condition")
    
    uploaded_file = st.file_uploader("Choose an image...", type=['bmp', 'BMP'])
    
    if uploaded_file is not None:
        with st.spinner('Analyzing image...'):
            results = analyze_image(uploaded_file)
            
            if results:
                # Get risk assessment
                risk_assessment = assess_dry_eye_risk(results['predicted_class'])
                
                # Create tabs for different sections
                tab1, tab2, tab3 = st.tabs(["Analysis Results", "Model Explanation", "Risk Assessment"])
                
                with tab1:
                    # Display original image and ROI
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Original Image with ROI")
                        original_with_roi = results['original_image'].copy()
                        x, y, w, h = results['roi_coords']
                        cv2.rectangle(original_with_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        st.image(cv2.cvtColor(original_with_roi, cv2.COLOR_BGR2RGB))
                    
                    with col2:
                        st.subheader("Extracted ROI")
                        st.image(results['roi'])
                        st.metric("Prediction", results['predicted_class'])
                        st.progress(float(results['prediction_prob']) / 100)
                        st.write(f"Confidence: {results['prediction_prob']:.2f}%")
                
                with tab2:
                    # LIME explanations
                    st.subheader("Model Explanations")
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        st.write("Positive Influences")
                        temp, mask = results['explanation'].get_image_and_mask(
                            results['explanation'].top_labels[0],
                            positive_only=True,
                            num_features=5,
                            hide_rest=True
                        )
                        # Create and display figure for positive influences
                        fig_pos = plot_lime_explanation(temp, mask, "Positive Influences")
                        st.pyplot(fig_pos)
                        plt.close(fig_pos)  # Clean up the figure
                    
                    with col4:
                        st.write("All Influences")
                        temp, mask = results['explanation'].get_image_and_mask(
                            results['explanation'].top_labels[0],
                            positive_only=False,
                            num_features=5,
                            hide_rest=False
                        )
                        # Create and display figure for all influences
                        fig_all = plot_lime_explanation(temp, mask, "All Influences")
                        st.pyplot(fig_all)
                        plt.close(fig_all)  # Clean up the figure
                    
                    # Feature importance
                    st.subheader("Feature Importance")
                    ind = results['explanation'].top_labels[0]
                    dict_heatmap = dict(results['explanation'].local_exp[ind])
                    
                    # Create feature importance visualization using matplotlib
                    importance_data = sorted(dict_heatmap.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                    segments, importances = zip(*importance_data)
                    
                    fig_importance, ax_importance = plt.subplots(figsize=(10, 6))
                    bars = ax_importance.bar(range(len(segments)), 
                                          [abs(imp) for imp in importances],
                                          color=['red' if imp < 0 else 'blue' for imp in importances])
                    ax_importance.set_xticks(range(len(segments)))
                    ax_importance.set_xticklabels([f'Segment {s}' for s in segments], rotation=45)
                    ax_importance.set_ylabel('Feature Importance')
                    ax_importance.set_title('Top 5 Most Influential Features')
                    
                    # Add value labels on top of each bar
                    for bar in bars:
                        height = bar.get_height()
                        ax_importance.text(bar.get_x() + bar.get_width()/2., height,
                                        f'{height:.3f}',
                                        ha='center', va='bottom')
                    
                    plt.tight_layout()
                    st.pyplot(fig_importance)
                    plt.close(fig_importance)
                
                with tab3:
                    # Risk Assessment Display
                    st.subheader("Dry Eye Risk Assessment")
                    
                    # Risk Level Alert
                    if risk_assessment["is_high_risk"]:
                        st.error(f"Risk Level: {risk_assessment['risk_level']}")
                    else:
                        st.success(f"Risk Level: {risk_assessment['risk_level']}")
                    
                    # Pattern Description
                    st.info("Pattern Analysis")
                    st.write(risk_assessment["pattern_description"])
                    
                    # Recommendations
                    st.subheader("Recommendations")
                    for rec in risk_assessment["recommendations"]:
                        st.write(f"• {rec}")
                    
                    # Additional Information Box
                    with st.expander("About Tear Film Patterns"):
                        st.write("""
                        Tear film patterns are crucial indicators of eye health:
                        - CO (Color): Indicates potential dry eye condition
                        - AM (Amorphous): Suggests stable tear film
                        - FL (Flow): Shows normal tear distribution
                        - MA (Marbled): Indicates good lipid layer
                        - MC (Meshwork): Suggests healthy tear film
                        """)

if __name__ == "__main__":
    main()
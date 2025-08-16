import streamlit as st
import joblib
import numpy as np
import shap
import plotly.graph_objects as go

# Load the trained model
model = joblib.load("random_forest_model.joblib")

# Define the label encoder classes
label_encoder_classes = ['Diabetes', 'Non-Diabetes', 'Pre-Diabetes']

st.title("Diabetes Status Prediction with Explanation")

st.write("Enter the patient's details to predict their diabetes status, see the probabilities, and understand the key factors influencing the prediction.")

# Input fields for the features used by the model
age = st.slider("Age", 18, 100, 50)
fbs = st.slider("FBS (Fasting Blood Sugar)", 50.0, 300.0, 100.0)
bmi = st.slider("BMI (Body Mass Index)", 15.0, 50.0, 25.0)
wc = st.slider("wc (Waist Circumference)", 50.0, 150.0, 90.0)
hc = st.slider("Hc (Hip Circumference)", 70.0, 150.0, 100.0)

# Create a numpy array from the input values
features = np.array([[fbs, bmi, age, wc, hc]])
feature_names = ['FBS', 'BMI', 'Age', 'wc', 'Hc'] # Add feature names

if st.button("Predict and Explain"):
    # Make prediction
    prediction_encoded = model.predict(features)
    predicted_status = label_encoder_classes[prediction_encoded[0]]

    # Get prediction probabilities
    prediction_proba = model.predict_proba(features)

    st.subheader("Prediction Results:")
    st.success(f"Predicted Diabetes Status: **{predicted_status}**")

    st.subheader("Prediction Probabilities:")
    for i, class_name in enumerate(label_encoder_classes):
        st.write(f"Probability of {class_name}: **{prediction_proba[0][i]:.2f}**")

    st.subheader("Explanation:")

    # Create a SHAP explainer object
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values for the single input instance
    shap_values = explainer.shap_values(features)

    # Access the SHAP values for the predicted class for the single instance
    try:
        # Attempt standard access for multi-class list output
        shap_values_instance = shap_values[prediction_encoded[0]][0]
    except (IndexError, TypeError):
        try:
            # Attempt access for multi-class array output (n_instances, n_classes, n_features)
            shap_values_instance = shap_values[0, prediction_encoded[0], :]
        except (IndexError, TypeError) as e:
            st.error(f"Error accessing SHAP values for explanation. Details: {e}")
            shap_values_instance = None # Set to None if access fails


    if shap_values_instance is not None:
        st.write(f"Here's what influenced the prediction of **{predicted_status}**: ")

        # Calculate absolute SHAP values for the predicted class and their sum
        abs_shap_values_predicted_class = np.abs(shap_values_instance)
        total_abs_shap_predicted_class = np.sum(abs_shap_values_predicted_class)

        if total_abs_shap_predicted_class > 0:
            # Calculate percentage contribution based on absolute SHAP values for the predicted class
            percentage_contributions = (abs_shap_values_predicted_class / total_abs_shap_predicted_class) * 100

            # Create a list of (feature_name, percentage) tuples
            feature_percentage_pairs = list(zip(feature_names, percentage_contributions))

            # Sort by percentage contribution
            feature_percentage_pairs.sort(key=lambda item: item[1], reverse=True)

            # Create labels and values for the donut chart
            labels = [f"{feature} ({percentage:.1f}%)" for feature, percentage in feature_percentage_pairs]
            values = [percentage for _, percentage in feature_percentage_pairs]

            # Create the donut chart
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
            fig.update_layout(title_text="Percentage Contribution of Features to Prediction")
            st.plotly_chart(fig)
        else:
            st.write("Feature contributions could not be determined.")


    st.subheader("Health Tips (According to WHO):")
    if predicted_status == 'Diabetes':
        st.write("""
        **Tips for managing Diabetes:**
        *   Eat a healthy diet: Focus on fruits, vegetables, whole grains, lean protein, and low-fat dairy. Limit sugary drinks, processed foods, and saturated and trans fats.
        *   Be physically active: Aim for at least 150 minutes of moderate-intensity aerobic activity or 75 minutes of vigorous-intensity aerobic activity per week.
        *   Maintain a healthy weight: Losing even a small amount of weight can improve blood sugar control.
        *   Take medications as prescribed: If your doctor has prescribed medication for your diabetes, take it as directed.
        *   Monitor your blood sugar levels regularly: This will help you understand how your diet, activity, and medications affect your blood sugar.
        *   See your doctor regularly: Regular check-ups are important for monitoring your diabetes and preventing complications.
        """)
    elif predicted_status == 'Pre-Diabetes':
        st.write("""
        **Tips for preventing Diabetes (for Pre-Diabetes):**
        *   Lose weight if you are overweight: Losing 5-10% of your body weight can significantly reduce your risk of developing type 2 diabetes.
        *   Eat a healthy diet: Similar to managing diabetes, focus on nutrient-rich foods and limit unhealthy ones.
        *   Get regular physical activity: Aim for at least 150 minutes of moderate-intensity aerobic activity per week.
        *   Don't smoke: Smoking increases the risk of developing type 2 diabetes and other health problems.
        *   Manage stress: Stress can affect blood sugar levels. Find healthy ways to manage stress.
        """)
    else: # Non-Diabetes
         st.write("""
        **Tips for maintaining Non-Diabetes status:**
        *   Continue healthy eating habits: Maintain a balanced diet rich in fruits, vegetables, and whole grains.
        *   Stay physically active: Regular exercise is crucial for overall health and preventing many chronic diseases, including diabetes.
        *   Maintain a healthy weight: Keep your weight in a healthy range for your height and body type.
        *   Get enough sleep: Aim for 7-9 hours of quality sleep per night.
        *   Manage stress: Find healthy ways to cope with stress.
        *   Regular check-ups: See your doctor for regular check-ups to monitor your overall health.
        """)

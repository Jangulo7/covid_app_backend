# DetCOVID App

**Predictive Technology for Health Improvement**

DetCOVID App is a user-friendly web application that utilizes advanced **Convolutional Neural Networks (CNNs)** to classify chest X-ray images into three distinct categories:

- **HEALTHY**
- **COVID-19**
- **PNEUMONIA**

It also provides the **confidence percentage** for each prediction, allowing users to understand how certain the model is about the given classification.

---

## How It Works

Simply upload a chest X-ray image, and within seconds the model will:

- Analyze your image.
- Predict whether the condition shown is **Healthy**, **COVID-19**, or **Pneumonia**.
- Display the confidence level of this prediction.

**Note:**  
**This tool does NOT replace professional medical diagnosis. It is for educational and exploratory purposes only.**

---

## About the AI Model

The app employs a robust CNN-based neural network model (**ensemble_unified_model3b.keras**) trained and validated on extensive clinical chest X-ray datasets. It achieved excellent accuracy and reliable performance in test evaluations, demonstrating high sensitivity and specificity in distinguishing among **Healthy**, **COVID-19**, and **Pneumonia** categories.

- **Model Type:** CNN-based Ensemble Neural Network
- **Framework:** TensorFlow/Keras
- **Input:** Chest X-ray images resized to `224x224` pixels
- **Output:** Classification label and confidence score

---

## Getting Started (Local Installation)

To run the application locally on your computer, follow these simple steps:

### Step 1: Clone the repository
```bash
git clone https://github.com/Jangulo7/covid_app.git
cd covid_app

### Step 2: Install the required packages
pip install -r requirements.txt

### Step 3: Run the Streamlit app
streamlit run app.py

## Deployment

The application is deployed directly on Streamlit Cloud, allowing easy access and scalability.

## Repository Structure

covid_app/
│
├── app.py                 # Main Streamlit app
├── requirements.txt       # Dependencies
├── README.md              # Documentation (this file)
├── .gitignore             # Files excluded from GitHub
└── models/
    └── ensemble_unified_model3b.keras  # Trained CNN model

## Feedback

Your feedback is valuable! Feel free to use the feedback section within the app to share your experience or suggestions.

## License & Author

Author: Johanna
Date: March, 2025
License: MIT

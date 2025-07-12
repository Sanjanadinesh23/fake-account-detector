Fake Account & AI-Generated Image Detection Web App


This is a Flask-based web application that detects fake social media accounts using machine learning and identifies whether an uploaded image is AI-generated or real.

Features:

Fake Account Detection: Uses a trained Random Forest model to predict whether a social media account is real or fake based on uploaded CSV data.

AI Image Detection: Identifies if an image is AI-generated or authentic using simple image metadata and size analysis.

Web Interface: Upload and get results directly through a user-friendly Flask web interface.

Model Training Script: Easily retrain the fake account detection model using model_training.py.

How To clone:
1. Clone the Repository
git clone https://github.com/Sanjanadinesh23/fake-account-detector.git
cd my_flask-app
2. Create Virtual Environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies
If requirements.txt has encoding issues, you may manually install key packages:
pip install flask pandas scikit-learn pillow pytesseract matplotlib joblib shap
4. Run the Flask App:
python app.py

NOTE:
The AI-generated image detection is a placeholder based on image size and metadata. You can replace it with a deep learning-based model for production.
Tesseract OCR should be installed and accessible via system PATH for pytesseract to work.

Screenshots

<img width="1280" height="720" alt="image" src="https://github.com/user-attachments/assets/a07520cf-1819-4f1d-8bf3-a5746baf6043" />
<img width="1280" height="720" alt="image" src="https://github.com/user-attachments/assets/c3b75574-ddbc-4b3d-af5a-1f36f71ea3bd" />
<img width="1280" height="720" alt="image" src="https://github.com/user-attachments/assets/dab2d86b-ca9d-46c7-8608-81e6d442f845" />
<img width="1280" height="720" alt="image" src="https://github.com/user-attachments/assets/c9a380e9-b1aa-41ea-94c9-b04291a544e5" />
<img width="1280" height="720" alt="image" src="https://github.com/user-attachments/assets/72953fc6-898f-44f8-8382-8a818651b202" />
<img width="1280" height="720" alt="image" src="https://github.com/user-attachments/assets/156d0610-520e-448c-9c74-65fa3071d35a" />
<img width="1280" height="720" alt="image" src="https://github.com/user-attachments/assets/5e1f3ded-8b2d-4f82-a21d-4e84801d39f3" />
<img width="1280" height="720" alt="image" src="https://github.com/user-attachments/assets/cd884023-d331-4ed6-888c-34b164fc5ac0" />
<img width="1280" height="720" alt="image" src="https://github.com/user-attachments/assets/ab64ef1c-3b3b-4975-8409-4965b00e5d58" />






from flask import Flask, redirect, request, jsonify, render_template, url_for
import os
import shap
import joblib
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from PIL import Image
import pytesseract  # For OCR
import requests
from detect_ai import predict_ai_generated

# Initialize Flask app
app = Flask(__name__)

# Configurations for uploads
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit
app.config['UPLOAD_FOLDER'] = 'static' 

# API Keys for Google Custom Search
GOOGLE_API_KEY = 'AIzaSyAiLrKNiqDGgPgFz9gkof6PZn2QAnA3u0Q'
GOOGLE_CX = '36ab03f8eaada44f8'

# Load the trained model for fake account detection
model = joblib.load('fake_account_model.pkl')

# Extract feature importances from the model
feature_importances = dict(zip(
    ["profile pic", "nums/length username", "fullname words", "nums/length fullname", 
     "name==username", "description length", "external URL", "private", 
     "#posts", "#followers", "#follows"],
    model.feature_importances_
))

feature_explanations = {
    "#followers": "Accounts with an unusually high or low number of followers are often considered suspicious.",
    "profile pic": "Accounts without a profile picture are more likely to be fake.",
    "fullname words": "Fake accounts sometimes use generic full names or a single word to avoid identification.",
    "#follows": "Fake accounts often follow an abnormally large number of other accounts.",
    "private": "Fake accounts are often private to hide their lack of genuine activity or followers.",
    "#posts": "Low posting activity can be a sign of an inactive or fake account.",
    "description length": "Short or overly generic descriptions can indicate fake or inactive accounts.",
    "nums/length fullname": "Full names containing excessive numbers are often associated with fake accounts or bots.",
    "name==username": "When the name matches the username exactly, it may indicate a lack of personalization, common in fake accounts.",
    "nums/length username": "A username with too many numbers or odd patterns is often associated with bots or fake accounts.",
    "external URL": "Accounts linking to questionable or suspicious external URLs may be fake."
}


# Function to convert "Yes"/"No" to 1/0
def convert_to_numeric(value):
    if value.lower() == "yes":
        return 1
    elif value.lower() == "no":
        return 0
    else:
        return value

# Function to extract text from an image using OCR
def extract_text(image_path):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

# Function to search Google using the Custom Search API
def search_google(query, num_results=5):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CX,
        "q": query,
        "searchType": "image",
        "num": num_results
    }
    response = requests.get(url, params=params)
    return response.json().get('items', [])

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def handle_login():
    # Example: Extract login credentials
    username = request.form['username']
    password = request.form['password']
    
    # Replace this with actual authentication logic
    if username == 'admin' and password == 'password':
        return redirect(url_for('home'))
    else:
        error_message = "Invalid username or password. Please try again."
        return render_template('login.html', error=error_message)

# @app.route('/home')
# def home():
#     return render_template('index.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/index1')
def image_detection():
    return render_template('index1.html')

@app.route('/index2')
def reverse_image_search():
    return render_template('index2.html')



import matplotlib.pyplot as plt
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON data from the request
        data = request.json
        print(f"Received data: {data}")

        # Convert JSON data to DataFrame
        df = pd.DataFrame([data])

        # Convert 'Yes'/'No' to 1/0 for boolean fields
        boolean_columns = ["profile pic", "name==username", "external URL", "private"]
        for col in boolean_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: 1 if str(x).lower() == "yes" else 0)

        # Ensure all numeric columns are properly converted
        numeric_columns = [
            "profile pic",
            "#posts",
            "name==username",
            "fullname words",
            "description length",
            "private",
            "nums/length fullname",
            "nums/length username",
            "#followers",
            "external URL",
            "#follows"
        ]


        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

        # Fill any NaN values with 0 (fallback for missing data)
        df = df.fillna(0)

        print(f"Processed DataFrame: {df}")

        # Predict using the model
        prediction = model.predict(df)[0]
        print(f"Prediction: {prediction}")

        if prediction == 1:  # Fake account
            try:
                # SHAP explainer for feature importance
                print('Starting SHAP calculations...')
                explainer = shap.TreeExplainer(model)  # Adjust based on model type
                shap_values = explainer.shap_values(df)
                print('SHAP values calculated successfully.')

                # Handle binary classification SHAP values (list for positive/negative class)
                if isinstance(shap_values, list):
                    if len(shap_values) > 1:
                        shap_values_for_prediction = shap_values[1]  # Positive class SHAP values
                    else:
                        shap_values_for_prediction = shap_values[0]
                else:
                    shap_values_for_prediction = shap_values

                print(f"SHAP values structure: {type(shap_values_for_prediction)}, shape: {shap_values_for_prediction.shape}")

                # Ensure SHAP values match input dimensions
                if len(shap_values_for_prediction[0]) != len(df.columns):
                    raise ValueError("Mismatch between SHAP values and input features. "
                                    f"Expected {len(df.columns)}, but got {len(shap_values_for_prediction[0])}.")

                # Extract SHAP values for the current prediction
                shap_values_for_row = shap_values_for_prediction[0]  # Extract values for the first row
                # Extract the first column and convert to 1D array
                shap_values_for_row = shap_values_for_row[:, 0]
                # Take absolute values of all shap_values
                shap_values_for_row = abs(shap_values_for_row)
                print(f"SHAP values for row: {shap_values_for_row}")

                # Combine features, SHAP values, and explanations
                feature_contributions = pd.DataFrame({
                    "feature": df.columns,
                    "shap_value": shap_values_for_row,
                    "explanation": [
                        feature_explanations.get(feature, "No explanation available.") for feature in df.columns
                    ]
                })

                # Sort by absolute SHAP value and take the top 5 features
                top_features = feature_contributions.sort_values(
                    by="shap_value", key=lambda x: abs(x), ascending=False
                ).head(2)
                print(f"Top features for explanation: {top_features}")

                # Generate SHAP values bar chart
                plt.figure(figsize=(10, 6))
                plt.barh(top_features['feature'], top_features['shap_value'], color='skyblue')
                plt.title("Local Feature Importance (SHAP Values)")
                plt.xlabel("SHAP Value")
                plt.ylabel("Feature")
                plt.gca().invert_yaxis()

                # Save the graph
                graph_filename = f"shap_importance_{hash(str(data))}.png"
                graph_path = os.path.join(app.config.get('UPLOAD_FOLDER', 'static'), graph_filename)
                plt.savefig(graph_path)
                plt.close()

                print(f"Graph saved at {graph_path}")

                print('-'*50)   # To print a division line

                # Return prediction result with explanations and graph URL
                return jsonify({
                    'prediction': 'Fake',
                    'explanation': top_features.to_dict(orient='records'),
                    'graph': url_for('static', filename=graph_filename)
                })

            except Exception as shap_error:
                print(f"SHAP Error: {shap_error}")
                return jsonify({
                    'prediction': 'Fake',
                    'error': f'Failed to calculate SHAP values or generate the graph. Details: {shap_error}'
                })
        else:
            return jsonify({'prediction': 'Real'})

    except Exception as main_error:
        print(f"Prediction Error: {main_error}")
        return jsonify({'error': 'An unexpected error occurred during prediction.'})










@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # AI Detection
    ai_detection_result = predict_ai_generated(file_path)

    return jsonify({
        "ai_detection": ai_detection_result
    })

@app.route('/index2', methods=['GET', 'POST'])
def image_search():
    if request.method == 'POST':
        uploaded_file = request.files['image']
        if uploaded_file:
            file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
            uploaded_file.save(file_path)

            # Extract text from the image
            query = extract_text(file_path)
            print(f"Extracted Query: {query}")

            if not query:
                query = "man"  # Fallback query if no text is found
                print("Fallback query used!")

            # Perform searches
            google_results = search_google(query)

            # Combine results
            results = {
                "google": [{"title": item.get('title', ''), "link": item.get('link', '')} for item in google_results]
            }

            # Clean up uploaded image
            os.remove(file_path)

            return render_template('results.html', results=results, query=query)

    return render_template('index2.html')

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)

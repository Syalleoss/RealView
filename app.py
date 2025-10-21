# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
from collections import Counter

app = Flask(__name__)
CORS(app)

# --- Load your models ---
try:
    model = joblib.load('logistic_model.pkl')    # Supervised model (e.g., LogisticRegression)
    kmeans = joblib.load('kmeans_model.pkl')      # Unsupervised model (e.g., KMeans)
    vectorizer = joblib.load('vectorizer.pkl')    # Feature extractor (e.g., TfidfVectorizer)
    print("Models loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    print("Please ensure 'logistic_model.pkl', 'kmeans_model.pkl', and 'vectorizer.pkl' are in the same directory as app.py.")
    model = None
    kmeans = None
    vectorizer = None
    print("FATAL ERROR: Models not loaded. Cannot perform accurate predictions.")
except Exception as e:
    print(f"An unexpected error occurred during model loading: {e}")
    model = None
    kmeans = None
    vectorizer = None
    print("FATAL ERROR: Models not loaded due to unexpected error. Cannot perform accurate predictions.")

# --- IMPORTANT: If your vectorizer requires text cleaning before transformation,
# --- define your clean_text function here, identical to how it was used in Colab.
# For example:
# import re
# def clean_text(text):
#     # Implement your text cleaning steps here (e.g., lowercase, remove punctuation, stopwords, stemming/lemmatization)
#     text = text.lower()
#     text = re.sub(r'[^a-z0-9\s]', '', text) # Remove non-alphanumeric
#     # Add more cleaning steps as per your model's training
#     return text


@app.route('/')
def home():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/predict_review', methods=['POST'])
def predict_review():
    """
    API endpoint to receive a review, process it, and return a prediction.
    Expects a JSON payload with 'review_text'.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    review = data.get('review_text')

    if not review:
        return jsonify({"error": "No 'review_text' provided in the request"}), 400

    # Initialize all prediction details
    supervised_prediction_label = "N/A" # Renamed to match user's Colab code for clarity
    supervised_confidence_real = "N/A"
    supervised_confidence_fake = "N/A"
    unsupervised_prediction_label = "N/A" # Renamed for clarity
    unsupervised_confidence_real = "N/A" 
    unsupervised_confidence_fake = "N/A" 
    final_verdict = "Analysis Pending"
    red_flags_list = []
    authenticity_score = 0 # Overall score for the progress bar

    # --- Check if models are loaded before attempting predictions ---
    if model is None or vectorizer is None or kmeans is None:
        error_message = "Backend models are not loaded. Cannot perform accurate prediction. Check server logs for details."
        print(error_message)
        return jsonify({
            "error": error_message,
            "supervised_prediction": "Error",
            "unsupervised_prediction": "Error",
            "final_verdict": "Error - Models not loaded",
            "authenticity_score": 0,
            "red_flags": ["Server models failed to load. Please contact support or check server logs."]
        }), 500

    # --- Models are loaded, proceed with actual predictions ---
    try:
        # Preprocess the review text:
        # UNCOMMENT THE LINE BELOW AND ENSURE `clean_text` IS DEFINED/IMPORTED
        # if your vectorizer was trained on cleaned text.
        # review_processed = clean_text(review)
        review_processed = review # Default to raw review if clean_text is not used

        vectorized_review = vectorizer.transform([review_processed])
        
        # Ensure vectorized_review is not empty or all zeros
        if vectorized_review.nnz == 0:
             return jsonify({"error": "Review could not be vectorized effectively. Too short or no relevant terms for your vectorizer."}), 400
        
        vec_array = vectorized_review.toarray() # Convert sparse to dense for numpy ops

        # --- Supervised Model Prediction (from user's accurate Colab snippet) ---
        supervised_proba = model.predict_proba(vectorized_review)[0]
        supervised_prediction_output = model.predict(vectorized_review)[0]

        # BASED ON USER'S COLAB CODE: supervised_proba[1] is Real, supervised_proba[0] is Fake
        supervised_confidence_real = round(supervised_proba[1] * 100, 2)
        supervised_confidence_fake = round(supervised_proba[0] * 100, 2)
        
        # BASED ON USER'S COLAB CODE: supervised_prediction_output == 1 is "Original" (Real), 0 is "Fake"
        supervised_prediction_label = "Original" if supervised_prediction_output == 1 else "Fake"

        # --- Unsupervised Model Prediction (from user's accurate Colab snippet) ---
        distances = [np.linalg.norm(vec_array - center) for center in kmeans.cluster_centers_]
        
        # Handle potential division by zero if a distance is exactly zero
        inverse_distances = [1 / d if d != 0 else 1e6 for d in distances] # Use 1e6 for d=0 to represent very close
        
        total_inverse = sum(inverse_distances)
        
        # Ensure total_inverse is not zero to prevent division by zero
        if total_inverse == 0:
            # This would imply all distances were non-zero but their inverses summed to zero, or similar issue.
            # Fallback to N/A or default.
            unsupervised_confidence_real = "N/A"
            unsupervised_confidence_fake = "N/A"
        else:
            # BASED ON USER'S COLAB CODE: inverse_distances[0] is Fake, inverse_distances[1] is Original
            unsupervised_confidence_fake = round((inverse_distances[0] / total_inverse) * 100, 2)
            unsupervised_confidence_real = round((inverse_distances[1] / total_inverse) * 100, 2)
        
        unsupervised_prediction_output = kmeans.predict(vectorized_review)[0]
        # BASED ON USER'S COLAB CODE: unsupervised_prediction_output == 0 is "Fake", 1 is "Original"
        unsupervised_prediction_label = "Fake" if unsupervised_prediction_output == 0 else "Original"

        # --- Final Verdict Logic (from user's accurate Colab snippet) ---
        # Note: This logic assumes supervised_prediction_output: 1=Original/Real, 0=Fake
        # And unsupervised_prediction_output: 1=Original, 0=Fake
        if supervised_prediction_output == 1 and unsupervised_prediction_output == 1:
            final_verdict = "✔️ Likely an Original Review"
        elif supervised_prediction_output == 0 or unsupervised_prediction_output == 0:
            final_verdict = "⚠️ Possible Fake Review"
        else:
            final_verdict = "❓ Uncertain—Further Analysis Needed"

        # --- Calculate overall authenticity score for progress bar ---
        # Use average of real confidences if available, or base on final verdict
        real_confs = []
        if isinstance(supervised_confidence_real, (int, float)):
            real_confs.append(supervised_confidence_real)
        if isinstance(unsupervised_confidence_real, (int, float)):
            real_confs.append(unsupervised_confidence_real)
        
        if real_confs:
            authenticity_score = int(np.clip(np.mean(real_confs), 0, 100))
        else:
            # Fallback for authenticity_score if no numerical confidences are available
            if "Original" in final_verdict:
                authenticity_score = 90
            elif "Fake" in final_verdict:
                authenticity_score = 10
            else:
                authenticity_score = 50 # Uncertain

        # --- Refine Red Flags based on final verdict and review content ---
        red_flags_list = [] # Always start fresh
        if "Fake" in final_verdict or "Uncertain" in final_verdict: # Check if the verdict suggests non-authentic or uncertain
            # Common patterns in fake reviews
            if len(review.split()) < 10:
                red_flags_list.append("Very short review text")
            if len(review.split()) > 50 and review.count('.') < 2:
                red_flags_list.append("Unusually long review lacking proper sentence breaks")
            
            if review.count('!') > 2 or review.count('?') > 2 or review.count('!!!') > 0 or review.count('???') > 0:
                red_flags_list.append("Excessive use of punctuation")
            
            common_superlatives = ["amazing", "best", "perfect", "excellent", "greatest", "awesome", "fantastic", "wonderful"]
            if any(word in review.lower() for word in common_superlatives):
                red_flags_list.append("Contains generic marketing or superlative words")
            
            generic_phrases = ["highly recommend", "satisfied customer", "good product", "works well", "must buy"]
            if any(phrase in review.lower() for phrase in generic_phrases):
                red_flags_list.append("Uses generic phrases without specific details")

            words = [word for word in review.lower().split() if word.isalpha() and len(word) > 2]
            word_counts = Counter(words)
            if any(count > 2 for word, count in word_counts.items()):
                red_flags_list.append("Unnatural word repetition detected")
            
            if not red_flags_list: 
                red_flags_list.append("Subtle linguistic or behavioral patterns detected by AI")
                
    except Exception as e:
        print(f"Error during prediction processing: {e}")
        return jsonify({"error": f"An error occurred during prediction processing on the server: {str(e)}"}), 500

    # Return the detailed results as JSON
    return jsonify({
        "supervised_prediction": supervised_prediction_label,
        "supervised_confidence_real": supervised_confidence_real,
        "supervised_confidence_fake": supervised_confidence_fake,
        "unsupervised_prediction": unsupervised_prediction_label,
        "unsupervised_confidence_real": unsupervised_confidence_real,
        "unsupervised_confidence_fake": unsupervised_confidence_fake,
        "final_verdict": final_verdict,
        "authenticity_score": authenticity_score,
        "red_flags": red_flags_list
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

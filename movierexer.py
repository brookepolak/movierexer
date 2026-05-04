from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from openai import OpenAI  # Groq uses OpenAI-compatible API
import numpy as np
import json
import os

# Load environment variables from .env file (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use system environment variables

app = Flask(__name__)
CORS(app)

# Initialize Groq client (FREE!)
# Get your free API key at: https://console.groq.com/keys
# IMPORTANT: Set GROQ_API_KEY as an environment variable
# Never commit your actual API key to GitHub!
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY environment variable not set. "
        "Get your free API key at: https://console.groq.com/keys"
    )

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

# Define your feature schema
FEATURE_SCHEMA = {
    "genres": ["thriller", "sci-fi", "drama", "horror", "comedy", "action", "romance"],
    "eras": ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"],
    "themes": ["psychological", "mind-bending", "indie", "cerebral", "experimental"]
}

@app.route('/')
def index():
    return render_template('name.html')

@app.route('/parse-movie-preferences', methods=['POST'])
def parse_preferences():
    try:
        data = request.get_json()
        user_input = data.get('input', '')
        
        if not user_input:
            return jsonify({"error": "No input provided"}), 400
        
        # Call Groq API (free!)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Fast and free model
            messages=[{
                "role": "system",
                "content": """You are a movie preference parser. Extract structured information from user input.
                Return ONLY a JSON object with these fields:
                - mentioned_movies: array of movie titles
                - mentioned_directors: array of director names
                - genres: array of genres from this list: thriller, sci-fi, drama, horror, comedy, action, romance
                - themes: array of themes from this list: psychological, mind-bending, indie, cerebral, experimental
                - eras: array of decades like ["2010s", "2020s"] from this list: 1960s, 1970s, 1980s, 1990s, 2000s, 2010s, 2020s. Can be empty, one, or multiple decades.
                
                IMPORTANT: When user says they DON'T like something (e.g., "I don't like old movies"), infer the OPPOSITE eras they DO like.
                - "don't like old movies" = ["2010s", "2020s"]
                - "not a fan of recent stuff" = ["1960s", "1970s", "1980s", "1990s", "2000s"]
                
                Example input: "I like movies like Coherence and Primer"
                Example output: {"mentioned_movies": ["Coherence", "Primer"], "mentioned_directors": [], "genres": ["sci-fi", "thriller"], "themes": ["psychological", "mind-bending", "cerebral"], "eras": ["2010s"]}
                
                Example input: "I don't like old movies"
                Example output: {"mentioned_movies": [], "mentioned_directors": [], "genres": [], "themes": [], "eras": ["2010s", "2020s"]}"""
            }, {
                "role": "user",
                "content": user_input
            }],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        # Parse JSON response
        parsed_data = json.loads(response.choices[0].message.content)
        
        # Convert to feature vector
        feature_vector = create_feature_vector(parsed_data)
        
        return jsonify({
            "success": True,
            "parsed": parsed_data,
            "feature_vector": feature_vector.tolist(),
            "feature_schema": FEATURE_SCHEMA
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def create_feature_vector(parsed_data):
    """Convert parsed data to numerical feature vector"""
    vector = []
    
    # Genre encoding (multi-hot)
    for genre in FEATURE_SCHEMA["genres"]:
        vector.append(1 if genre in parsed_data.get("genres", []) else 0)
    
    # Theme encoding (multi-hot)
    for theme in FEATURE_SCHEMA["themes"]:
        vector.append(1 if theme in parsed_data.get("themes", []) else 0)
    
    # Era encoding (multi-hot) - CHANGED FROM ONE-HOT
    # Now supports multiple eras like ["2010s", "2020s"]
    for era in FEATURE_SCHEMA["eras"]:
        vector.append(1 if era in parsed_data.get("eras", []) else 0)
    
    # Add counts as features
    vector.append(len(parsed_data.get("mentioned_movies", [])))
    vector.append(len(parsed_data.get("mentioned_directors", [])))
    
    return np.array(vector)

if __name__ == '__main__':
    app.run(debug=True)
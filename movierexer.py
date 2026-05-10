from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from openai import OpenAI
import numpy as np
import json
import os
from reddit_recs import get_recommendations

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__)
CORS(app)

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable not set.")

client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

FEATURE_SCHEMA = {
    "genres": ["thriller", "sci-fi", "drama", "horror", "comedy", "action", "romance"],
    "eras":   ["1920s","1930s","1940s","1950s","1960s","1970s","1980s","1990s","2000s","2010s","2020s"],
    "themes": ["psychological", "mind-bending", "indie", "cerebral", "experimental"]
}

@app.route('/')
def index():
    return render_template('name.html')

@app.route('/parse-movie-preferences', methods=['POST'])
def parse_preferences():
    try:
        data       = request.get_json()
        user_input = data.get('input', '')

        if not user_input:
            return jsonify({"error": "No input provided"}), 400

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "system",
                "content": """You are a movie preference parser. Extract structured information from user input.
                Return ONLY a JSON object with these fields:
                - mentioned_movies: array of movie titles
                - mentioned_directors: array of director names, and also directors of any movies listed
                - genres: array of genres from this list: thriller, sci-fi, drama, horror, comedy, action, romance
                - themes: array of themes from this list: psychological, mind-bending, indie, cerebral, experimental
                - eras: array of decades like ["2010s", "2020s"] from this list: 1960s, 1970s, 1980s, 1990s, 2000s, 2010s, 2020s.

                IMPORTANT: When user says they DON'T like something (e.g., "I don't like old movies"), infer the OPPOSITE eras they DO like.
                - "don't like old movies" = ["2010s", "2020s"]
                - "not a fan of recent stuff" = ["1960s", "1970s", "1980s", "1990s", "2000s"]

                Example input: "I like movies like Coherence and Primer"
                Example output: {"mentioned_movies": ["Coherence", "Primer"], "mentioned_directors": [], "genres": ["sci-fi", "thriller"], "themes": ["psychological", "mind-bending", "cerebral"], "eras": ["2010s"]}"""
            }, {
                "role": "user",
                "content": user_input
            }],
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        parsed_data    = json.loads(response.choices[0].message.content)
        feature_vector = create_feature_vector(parsed_data)

        return jsonify({
            "success":        True,
            "parsed":         parsed_data,
            "feature_vector": feature_vector.tolist(),
            "feature_schema": FEATURE_SCHEMA
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get-recs', methods=['POST'])
def get_recs():
    """
    Takes mentioned_movies from the parsed preferences,
    searches Reddit via Arctic Shift, and returns movie recommendations
    with TMDB posters and Rotten Tomatoes links.
    """
    try:
        data            = request.get_json()
        mentioned       = data.get('mentioned_movies', [])

        if not mentioned:
            return jsonify({"success": True, "recommendations": []})

        genres          = data.get('genres', [])
        recommendations = get_recommendations(mentioned, genres=genres, max_recs=120)

        return jsonify({
            "success":         True,
            "recommendations": recommendations
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def create_feature_vector(parsed_data):
    vector = []
    for genre in FEATURE_SCHEMA["genres"]:
        vector.append(1 if genre in parsed_data.get("genres", []) else 0)
    for theme in FEATURE_SCHEMA["themes"]:
        vector.append(1 if theme in parsed_data.get("themes", []) else 0)
    for era in FEATURE_SCHEMA["eras"]:
        vector.append(1 if era in parsed_data.get("eras", []) else 0)
    vector.append(len(parsed_data.get("mentioned_movies", [])))
    vector.append(len(parsed_data.get("mentioned_directors", [])))
    return np.array(vector)


if __name__ == '__main__':
    app.run(debug=True)
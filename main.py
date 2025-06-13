from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import fitz  # PyMuPDF
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load and extract resume text from PDF
def extract_resume_text(pdf_path="resume.pdf"):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error reading resume.pdf: {e}")
        return "Resume data is currently unavailable."

resume_text = extract_resume_text()

@app.route("/")
def index():
    return "Resume chat API is running."

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a helpful assistant trained on Braden Bradshaw's resume. Here is the resume:\n{resume_text}"
                },
                {"role": "user", "content": user_input}
            ]
        )
        return jsonify({"response": response.choices[0].message.content})

    except Exception as e:
        print(f"Error during OpenAI chat completion: {e}")
        return jsonify({"error": "Server error"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=10000, host="0.0.0.0")

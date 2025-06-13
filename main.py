from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import openai
import fitz  # PyMuPDF

app = Flask(__name__)
CORS(app)

openai.api_key = os.getenv("OPENAI_API_KEY")

# Load resume/CV/publications text from PDF- I want to clean this up later and make it a little more token-efficient
def extract_resume_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

RESUME_TEXT = extract_resume_text("resume.pdf")

@app.route("/")
def home():
    return "Resume Chat Backend is live!"

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant trained on Braden Bradshaw's resume. "
                    "Reference only the resume when responding to questions about Braden's experience, skills, or background. "
                    "If someone asks for contact information, only provide it if explicitly requested. "
                    f"Here is the resume:\n\n{RESUME_TEXT}"
                )
            },
            {"role": "user", "content": user_input}
        ]
    )

    return jsonify({"reply": response.choices[0].message["content"]})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

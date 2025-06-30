from transformers import pipeline
from flask import Flask, request, jsonify

app = Flask(__name__)
generator = pipeline("text2text-generation", model="t5-small")  # Or any LLM you prefer

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    result = generator(prompt, max_length=100, do_sample=True)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002)

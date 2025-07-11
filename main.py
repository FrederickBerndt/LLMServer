from transformers import pipeline
from flask import Flask, request, jsonify

app = Flask(__name__)
generator = pipeline("text2text-generation", model="google/flan-t5-large")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    result = generator(prompt, max_length=200, do_sample=True)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)

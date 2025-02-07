from flask import Flask, request, jsonify, render_template
from nlp_model.similarity import find_similar_answer

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")  # Renders the frontend HTML

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("query")
    response = find_similar_answer(user_input)  # Find answer using NLP
    return jsonify({"response": response})  # Return the response as JSON

if __name__ == "__main__":
    app.run(debug=True)

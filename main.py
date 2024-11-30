# this is the main program
import os
import openai

from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPEN_AI_API_KEY")

app = Flask(__name__)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "message field is required."}), 400
    return main(data), 200


def main(data):
    chat_history = [
        {"role": "system", "content": "You have absolutely no personality."},
        {"role": "user", "content": data["message"]}
    ]

    chat_response = openai.ChatCompletion.create(model="gpt-4o-mini", messages=chat_history)

    chat_message = chat_response.choices[0].message
    chat_history.append(chat_message)

    print(chat_message.content)
    return jsonify(chat_message)


if __name__ == "__main__":
    app.run()

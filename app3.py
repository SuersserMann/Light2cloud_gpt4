from flask import Flask, render_template, request
import openai

app = Flask(__name__)

# 使用你的 GPT-4 API key
openai.api_key = "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"

@app.route("/")
def index():
    return render_template("index3.html")

@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query")
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Search query: {query}\nAnswer:",
        temperature=0.5,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    answer = response.choices[0].text.strip()
    return answer

@app.route("/get_answer", methods=["POST"])
def get_answer():
    query = request.form.get("query")
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Search query: {query}\nAnswer:",
        temperature=0.5,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    answer = response.choices[0].text.strip()
    return answer

if __name__ == "__main__":
    app.run(debug=True)

import openai
from flask import Flask, render_template, request

openai.api_key = "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"


def ask_question(question, model):
    response = openai.Completion.create(
        engine=model,
        prompt=f"Question: {question}\nAnswer:",
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    answer = response.choices[0].text.strip()
    return answer

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        question = request.form['question']
        answer = ask_question(question, "text-davinci-003")
        return render_template('index1.html', question=question, answer=answer)
    else:
        return render_template('index1.html')

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request
import openai
import os

app = Flask(__name__)

# 设置 OpenAI API 密钥
openai.api_key = "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"

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # 保存上传的文件
        file = request.files["file"]
        filename = file.filename
        file.save(os.path.join("uploads", filename))

        # 使用 GPT 进行问答
        prompt = request.form["prompt"]
        with open(os.path.join("uploads", filename), "r",encoding='UTF-8') as f:
            text = f.read()
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"{text}\nQ:全文使用敬语，以尊敬的主人您好为开头{prompt}\nA:",
            temperature=0.7,
            max_tokens=1024,
            n=1,
            stop=None,
            timeout=10,
        )
        answer = response.choices[0].text.strip()

        return render_template("index2.html", answer=answer)

    return render_template("index2.html")
if __name__ == '__main__':
    app.run(debug=True)
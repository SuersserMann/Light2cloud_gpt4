import requests
import json
import openai
# 用你自己的API密钥替换这里的YOUR_API_KEY
API_KEY = "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"
openai.api_key = "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"
# OpenAI的GPT-4 API请求URL
URL = "https://api.openai.com/v1/engines/davinci-codex/completions"

# API请求头部
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def gpt4_api(question):
    prompt = f"{question}"
    data = {
        "prompt": prompt,
        "max_tokens": 100,
        "n": 1,
        "stop": None,
        "temperature": 1,
    }

    response = requests.post(URL, headers=headers, data=json.dumps(data))
    response_data = response.json()

    if response.status_code == 200:
        return response_data['choices'][0]['text'].strip()
    else:
        return "Error: API request failed."

# 示例：用GPT-4 API回答问题
question = "什么是人工智能？"
answer = gpt4_api(question)
print(answer)
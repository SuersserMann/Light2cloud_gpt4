import openai
openai.api_key = "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"
import os
os.environ["http_proxy"]="127.0.0.1:10794"
os.environ["https_proxy"]="127.0.0.1:10794"
try:
    models = openai.Model.list()
    print("API key is valid")
except Exception as e:
    print("Error:", e)

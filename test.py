import openai
import sys

def check_gpt_key(api_key):
    openai.api_key = api_key

    try:
        prompt = "GPT 密钥测试"
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=10,
            n=1,
            stop=None,
            temperature=0.5,
        )
        print("GPT 密钥有效!")
        return True
    except openai.error.AuthenticationError as e:
        print("GPT 密钥无效，请检查您的密钥。")
        return False
    except Exception as e:
        print("出现异常，请检查您的网络连接。")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法：python check_gpt_key.py YOUR_API_KEY")
        sys.exit(1)

    api_key = sys.argv[1]
    check_gpt_key(api_key)

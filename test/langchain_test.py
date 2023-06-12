from langchain.llms import OpenAI
import openai

llm = OpenAI(temperature=0.9)#model_name=???,
text = "What would be a good company name for a company that makes colorful socks?"
print(llm(text))

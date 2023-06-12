from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
#结合一些内容，输出一个完整表单给llm
llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
chain = LLMChain(llm=llm, prompt=prompt)#创建一个非常简单的链，它将接受用户输入，用它格式化提示，然后将它发送到 LLM
print(prompt.format(product="colorful socks"))
print(chain.run("colorful socks"))
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
import os
os.environ["http_proxy"]="127.0.0.1:10794"
os.environ["https_proxy"]="127.0.0.1:10794"

template = """

{history}
Human: {human_input}
Assistant:"""

prompt = PromptTemplate(
    input_variables=["history", "human_input"],#输入变量，history和human_input
    template=template                          #样本等于上面的样本，初始history为空，human——input为空
)

chatgpt_chain = LLMChain(
    llm=OpenAI(temperature=0),
    prompt=prompt,
    # verbose=True,
    memory=ConversationBufferWindowMemory(k=2),
    #保存最近的一次输入和输出到memory字典中
    #[  {    "human_input": "你好",    "generated_output": "你好，有什么我可以帮助你的吗？"  },
    # {    "human_input": "我想知道天气怎么样？",    "generated_output": "当前天气晴朗，气温为20摄氏度。"  }]
)
while True:
    user_input = input("you:")
    output = chatgpt_chain.predict(human_input=user_input)
    print(output)


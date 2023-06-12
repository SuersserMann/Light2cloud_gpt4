from langchain import OpenAI, ConversationChain

llm = OpenAI(temperature=0)
conversation = ConversationChain(llm=llm, verbose=True)

conversation.predict(input="今天周几")

conversation.predict(input="我的上一个问题是什么")
conversation.predict(input="我的上一个问题是什么")
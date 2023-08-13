import os
import sys
import gradio as gr
import random

import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

import constants


os.environ["OPENAI_API_KEY"] = constants.APIKEY
#openai_api_key = os.getenv("OPENAI_API_KEY")

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False

query = None
if len(sys.argv) > 1:
    query = sys.argv[1]

if PERSIST and os.path.exists("persist"):
    print("Reusing index...\n")
    vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    # loader = TextLoader("data/data.txt") # Use this line if you only need data.txt
    loader = DirectoryLoader("data/")
    if PERSIST:
        index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
    else:
        index = VectorstoreIndexCreator().from_loaders([loader])

chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

#chat_history = []

def generate_answer(query):
    if query.lower() == "关于我们":
        wisdom_statements = [
           "我们是深圳大学万物归一队，用物联网的力量改变社会。",
           "我们是来自深圳大学万物归一队的成员，运用物联网的力量让生活变得更加美好。"

        ]
        return random.choice(wisdom_statements)

    if query.lower() == "联系管理员":
        wisdom_statements = [
           "邮箱：；电话：；如遇到紧急情况请拨打110" ,
           "电子邮件：；电话号码：；如遇到紧急情况请拨打110，或及时就医"
        ]
        return random.choice(wisdom_statements)


        #global chat_history
    query = "你是一名医生，根据病人的症状推荐合适的药品： " + query
        #result = chain({"question": query, "chat_history": chat_history})
        #chat_history.append((query, result['answer']))
        #return result['answer']
        #global chat_history
        #result = chain({"question": query, "chat_history": chat_history})
    result = chain({"question": query, "chat_history": []})
        #chat_history.append((query, result['answer']))
    return result['answer']


   

inputs = gr.inputs.Textbox(lines=2, label="请输入你的症状")
outputs = gr.outputs.Textbox(label="人工智能推荐：")

title = "万象归一"
description = '药品智能柜为您推荐合适的药品，如遇到紧急情况请立刻就医，网络不好的时候回答会稍慢，请耐心等待'
examples = [
    ["关于我们"],
    ["联系管理员"]
]
             

iface = gr.Interface(fn=generate_answer, inputs= inputs, outputs=outputs, title=title, description=description, examples=examples)#.launch()#share=True
iface.launch()#inline = False
from langchain_core.messages import ChatMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from langchain_openai import ChatOpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langserve.pydantic_v1 import BaseModel, Field
from typing import List, Union

from openai import OpenAI
from typing import List

import streamlit as st
import numpy as np
import time
import os


RAG_PROMPT_TEMPLATE = "You always answer into Korean. You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"

SYS_PROMPT_TEMPLATE = "You are a helpful, smart, kind, and efficient AI assistant. You always fulfill the user's requests to the best of your ability. You always answer succinctly. You must always answer in Korean."


class MyEmbeddings(Embeddings):
    def __init__(self, base_url, api_key="lm-studio"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def embed_documents(self, texts: List[str], model="nomic-ai/nomic-embed-text-v1.5-GGUF") -> List[List[float]]:
        texts = list(map(lambda text:text.replace("\n", " "), texts))
        datas = self.client.embeddings.create(input=texts, model=model).data
        return list(map(lambda data:data.embedding, datas))
        
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


# class InputChat(BaseModel): # 변수이름 템플릿이랑 같게.
#     messsages1: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(
#         ...,
#         description="The chat messages representing the current conversation.",
#     )

# 
st.set_page_config(page_title="My Chat Bot")


# 언어모델
if 'llm' not in st.session_state:
    with st.spinner("Loading LLM..."):
        st.session_state['llm'] = ChatOpenAI(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
            temperature=0.1,
        )
llm = st.session_state['llm']


# 임베딩 모델
if 'emb' not in st.session_state:
    with st.spinner("Loading LLM..."):
        st.session_state['emb'] = MyEmbeddings(base_url="http://localhost:1234/v1")
emb = st.session_state['emb']


# 문서 분할
if 'splitter' not in st.session_state:
    st.session_state['splitter'] = RecursiveCharacterTextSplitter(
        chunk_size=200, chunk_overlap=50,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""], length_function=len,
    )
text_splitter = st.session_state['splitter']


# 체인
if 'rag_chain' not in st.session_state:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You always answer into Korean."),
        ("user", RAG_PROMPT_TEMPLATE),
    ])
    st.session_state['rag_chain'] = prompt | llm | StrOutputParser()


# 체인 2
if 'chat_chain' not in st.session_state:
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYS_PROMPT_TEMPLATE),
        MessagesPlaceholder(variable_name='messsages1'),
    ])
    st.session_state['chat_chain'] = prompt | llm | StrOutputParser()#.with_types(input_type=InputChat)

    
# 채팅 이력
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(role="assistant", content="앞에 '!'(느낌표)를 붙이면 문서검색 후 답변합니다."),
    ]


# 파일 -> 벡터저장소
@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    # 문서 불러오고 분할
    with open("./temp", 'wb') as f:
        f.write(file.read())
    Loader = {'txt':TextLoader, 'pdf':PyPDFLoader}[file.name.split('.')[-1].lower()]
    docs = Loader("./temp").load_and_split(text_splitter=text_splitter)
    os.remove("./temp")
    # 벡터화
    vectorstore = FAISS.from_documents(docs, embedding=emb, distance_strategy=DistanceStrategy.COSINE)
    retriever = vectorstore.as_retriever(search_kwargs={'k':10})
    return retriever


##########################################################################################

# 제목
st.title('RAG Chat Bot')

# 파일 업로드 위젯
with st.sidebar:
    file = st.file_uploader("파일 업로드", type=["pdf", "txt", ], )
if file: retriever = embed_file(file)


# 채팅 내역 출력
for msg in st.session_state['messages']:
    st.chat_message(msg.role).write(msg.content)

# 유저 입력
if user_input := st.chat_input():
    retrieve_flag = user_input[0] == '!'
        
    st.session_state['messages'].append(ChatMessage(role='user', content=user_input))
    st.chat_message('user').write(user_input)
    if retrieve_flag: user_input = user_input[1:]
    
    if file and retrieve_flag:
        format_docs = lambda docs:"\n\n".join(doc.page_content for doc in docs)
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | st.session_state['rag_chain']
        )
        chain_input = user_input
    else:
        chain = st.session_state['chat_chain']
        chain_input = st.session_state['messages']
        
    with st.chat_message('assistant'):
        bot_out = st.empty()
        msg = ''
        for t in chain.stream(chain_input):
            msg += t
            bot_out.markdown(msg)
        
    st.session_state['messages'].append(ChatMessage(role='assistant', content=msg))
    








import streamlit as st
from langchain.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from config import MAX_HISTORY, CONTEXT_SIZE
from prompt_template import TEMPLATE
from utils.memory import clear_memory

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="history"
    )

llm = ChatOllama(model="llama3.2", streaming=True)

prompt_template = PromptTemplate(
    input_variables=["query", "history"],
    template=TEMPLATE
)

chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    memory=st.session_state.memory
)

st.title("NLI Chatbot")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask something"):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""

        for chunk in chain.stream({"query": prompt}):
            if isinstance(chunk, dict) and "text" in chunk:
                text_chunk = chunk["text"]
                full_response += text_chunk
                response_container.markdown(full_response)

    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
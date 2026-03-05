import streamlit as st
from langchain.memory import ConversationBufferMemory

def clear_memory():
    st.session_state.chat_history = []
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="history"
    )
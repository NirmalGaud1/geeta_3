#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Configure models
genai.configure(api_key="AIzaSyA-9-lTQTWdNM43YdOXMQwGKDy0SrMwo6c")
gemini = genai.GenerativeModel('gemini-pro')
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Custom CSS for styling
st.markdown("""
<style>
    .stApp {
        background: #f8f5e6;
        background-image: radial-gradient(#d4d0c4 1px, transparent 1px);
        background-size: 20px 20px;
    }
    .gita-font {
        font-family: 'Times New Roman', serif;
        color: #2c5f2d;
    }
    .user-msg {
        background: #ffffff !important;
        border-radius: 15px !important;
        border: 2px solid #2c5f2d !important;
    }
    .bot-msg {
        background: #fff9e6 !important;
        border-radius: 15px !important;
        border: 2px solid #ffd700 !important;
    }
    .stChatInput {
        background: #ffffff;
    }
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('bhagwatgeeta.csv')
    df['context'] = df.apply(
        lambda row: f"Chapter {row['Chapter']}.{row['Verse']}: {row['EngMeaning']}", 
        axis=1
    )
    embeddings = embedder.encode(df['context'].tolist())
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
    return df, index

df, faiss_index = load_data()

# App Header
st.markdown('<h1 class="gita-font">🕉 Bhagavad Gita Wisdom & General Assistant</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="gita-font">Spiritual Guidance & Practical Knowledge</h3>', unsafe_allow_html=True)
st.markdown("---")

# Response Classifier
def is_gita_question(query):
    response = gemini.generate_content(
        f"Answer only 'yes' or 'no': Is this question about Bhagavad Gita, Hindu philosophy, or spirituality?\n{query}"
    )
    return "yes" in response.text.lower()

# RAG Response
def get_gita_answer(query):
    query_embedding = embedder.encode([query])
    _, I = faiss_index.search(query_embedding.astype('float32'), k=3)
    contexts = [df.iloc[i]['context'] for i in I[0]]
    
    prompt = f"""Answer strictly using Bhagavad Gita teachings:
    Contexts: {contexts}
    Question: {query}
    - Cite chapter.verse numbers
    - Include Sanskrit terms in brackets
    - Explain practical application
    """
    return gemini.generate_content(prompt).text

# General Response
def get_general_answer(query):
    return gemini.generate_content(f"Answer concisely: {query}").text

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], 
                        avatar="🙋" if message["role"] == "user" else "🕉️"):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Seeking wisdom..."):
        try:
            if is_gita_question(prompt):
                response = get_gita_answer(prompt)
                response = f"**Bhagavad Gita Perspective**:\n{response}"
            else:
                response = f"**General Knowledge**:\n{get_general_answer(prompt)}"
        except:
            response = "Please rephrase your question"
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()


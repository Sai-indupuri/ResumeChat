import os
import numpy as np
from langchain_openai.embeddings import OpenAIEmbeddings
import streamlit as st
# Initialize OpenAI Embeddings
OPENAI_API_KEY = st.secrets["openai"]["api_key"]
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

def generate_embedding(text):
    """
    Generate an embedding for the given text using OpenAI's API.
    """
    try:
        return embeddings.embed_documents([text])[0]
    except Exception as e:
        raise ValueError(f"Error generating embedding: {e}")



def cosine_similarity(embedding_a, embedding_b):
    """
    Compute the cosine similarity between two embedding vectors.
    """
    embedding_a = np.array(embedding_a)
    embedding_b = np.array(embedding_b)
    return np.dot(embedding_a, embedding_b) / (np.linalg.norm(embedding_a) * np.linalg.norm(embedding_b))

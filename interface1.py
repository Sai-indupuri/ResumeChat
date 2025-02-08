import json
import numpy as np
import pandas as pd
import faiss
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.schema import HumanMessage, AIMessage
from llm import generate_response_with_llm  

OPENAI_API_KEY = st.secrets["openai"]["api_key"]

# ✅ Load CSV Data
CSV_FILE = r".\ResumeChat\updated_with_embeddings.csv"
df = pd.read_csv(CSV_FILE)

# ✅ Convert JSON string embeddings back to lists
df["embeddings"] = df["embeddings"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)


# ✅ Streamlit App Configuration
st.set_page_config(page_title="RiseON Chatbot", layout="wide")
st.image(r'riseon.png', output_format='PNG', width=250)
st.title("RiseON Resume Screening")

# ✅ Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hi! How can I assist you?")]

if "matching_profiles" not in st.session_state:
    st.session_state.matching_profiles = []

if "profile_ids" not in st.session_state:
    st.session_state.profile_ids = df["profileid"].unique().tolist()  # Use all profile IDs from CSV

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)
            
# ✅ Sidebar Controls
with st.sidebar:
    st.button("Clear Chat", on_click=lambda: st.session_state.clear())
    st.markdown("### Example Queries")
    st.write("Try these:")
    st.markdown("""
    - "Share the candidates who are freshers."
    - "Who has an MBA and AI skills?"
    - "Candidates for web development roles."
    """)

    st.markdown("### About Us")
    st.write("Learn more about our platform:")
    st.markdown("""
    - [HappyPeopleAI](https://happypeopleai.com/)
    - [RiseON](https://riseon.happypeopleai.com/)
    """)

def prepare_context_for_llm(chat_history, max_context_messages=1):
    """
    Prepare a limited chat context for the LLM.
    """
    return chat_history[-max_context_messages:]


# ✅ Initialize Embeddings Model
embeddings_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

def retrieve_matching_profiles(user_query, top_k=10):
    """Retrieve profiles using Cosine Similarity only."""

    # 🔹 Convert Query to Embedding
    query_embedding = embeddings_model.embed_query(user_query)
    query_embedding = np.array(query_embedding).reshape(1, -1)

    # 🔹 Convert Stored Embeddings to NumPy Array
    stored_embeddings = np.array(df["embeddings"].tolist(), dtype=np.float32)

    # 🔹 Compute Cosine Similarity
    similarity_scores = cosine_similarity(query_embedding, stored_embeddings)[0]

    # 🔹 Attach Similarity Scores to Profiles
    df["similarity"] = similarity_scores

    # 🔹 Sort Profiles by Cosine Similarity & Select Top `top_k`
    sorted_profiles = df.sort_values(by="similarity", ascending=False).head(top_k)

    # 🔹 Return Required Fields
    return sorted_profiles[["name", "summary", "total_years_workex", "profile_url"]].to_dict(orient="records")

# ✅ User Input Handling
user_query = st.chat_input("Type your message...")

if user_query:
    with st.chat_message("Human"):
        st.write(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))

    # 🔹 Retrieve Profiles (Cosine Similarity Only)
    with st.chat_message("AI"):
        with st.spinner("Retrieving matching profiles..."):
            matching_profiles = retrieve_matching_profiles(user_query, top_k=10)

    # 🔹 Generate Response with LLM
    with st.spinner("Generating response..."):
        recent_history = prepare_context_for_llm(st.session_state.chat_history, max_context_messages=2)

        response_stream = generate_response_with_llm(
            user_query, profile_details=matching_profiles, chat_history=recent_history
        )

    streamed_response = st.write_stream(response_stream)
    st.session_state.chat_history.append(AIMessage(content=streamed_response))

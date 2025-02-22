import streamlit as st
from searches import *
from llm import extract_query_metadata, generate_answer
from langchain.schema import HumanMessage, AIMessage

# ✅ Streamlit UI Configuration
st.set_page_config(page_title="RiseON Chatbot", layout="wide")
st.image('riseon.png', output_format="PNG", width=250)
st.title("RiseON Resume Screening")

# ✅ Initialize Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hi! How can I assist you?")]

# ✅ Display Chat History
for message in st.session_state.chat_history:
    with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
        st.write(message.content)



github_url = "https://sai-indupuri.github.io/ResumeChat/profile_data.html"

st.sidebar.markdown(
    f"""
    <style>
    .open-button {{
        display: block;
        width: 100%;
        padding: 8px;
        font-size: 14px;
        font-weight: 500;
        background-color: white;
        color: black;
        border: 1px solid #d3d3d3;
        border-radius: 5px;
        text-align: center;
        text-decoration: none;
        transition: background-color 0.2s ease-in-out, color 0.2s ease-in-out;
    }}

    .open-button:hover {{
        background-color: #f0f2f6;
        color: black;
    }}
    </style>

    <a href="{github_url}" target="_blank" class="open-button">Open Profile Data</a>
    """,
    unsafe_allow_html=True
)




# ✅ Sidebar: Clear Chat & Example Queries
with st.sidebar:
    st.button("Clear Chat", on_click=lambda: st.session_state.clear())
    st.markdown("### Example Queries")
    st.write("- 'Show freshers with AI skills.'")
    st.write("- 'Find backend developers with AWS experience.'")
    st.write("- 'Who are the most experienced AI engineers?'")
    st.markdown("### About Us")
    st.write("- [HappyPeopleAI](https://happypeopleai.com/)")
    st.write("- [RiseON](https://riseon.happypeopleai.com/)")


def query_pipeline(user_query, top_k=10):
    """
    Full retrieval pipeline: 
    1. Classifies query using LLM.
    2. Routes query to Hybrid Search (Metadata + Vector) or Ranking Search.
    3. Returns the best-matched candidates.
    """

    # ✅ Step 1: Extract Query Metadata using LLM
    query_metadata = extract_query_metadata(user_query)
    print(f"Extracted Metadata: {query_metadata}")

    # Extract metadata filters
    min_experience = query_metadata.get("min_experience")
    max_experience = query_metadata.get("max_experience")
    job_role = query_metadata.get("job_role")
    skills = query_metadata.get("skills", [])
    ranking_type = query_metadata.get("ranking_type")

    # ✅ Step 2: Handle Ranking Queries (Most/Least Experienced)
    if ranking_type:
        top_profiles = ranking_search(ranking_type, min_experience, max_experience, job_role, skills, top_k)

    else:
        # ✅ Step 3: Handle Freshers Separately
        if min_experience == 0 and max_experience == 2:
            print("📌 Running Freshers Metadata Search...")
            top_profiles = metadata_search(min_experience, max_experience, job_role, top_k)

        else:
            # ✅ Step 4: Use Hybrid Search (Metadata + Vector)
            top_profiles = hybrid_search(user_query, min_experience, max_experience, job_role, skills, top_k)

        # 🔹 Fallback: If no profiles found via hybrid search, use full vector search
        if not top_profiles:
            print("⚠️ No profiles after hybrid search. Running vector search instead.")
            top_profiles = vector_search(user_query, top_k)

    # ✅ Step 5: Generate the final answer
    return generate_answer(user_query, top_profiles) if top_profiles else "No matching profiles found."




# ✅ User Query Handling
user_query = st.chat_input("Type your message...")

if user_query:
    with st.chat_message("Human"):
        st.write(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("AI"):
        with st.spinner("Generating response..."):
            response = query_pipeline(user_query, top_k=15)
        streamed_response = st.write_stream(response)
        st.session_state.chat_history.append(AIMessage(content=streamed_response))

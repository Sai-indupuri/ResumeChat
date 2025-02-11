import os
import webbrowser
import time
import pandas as pd
import streamlit as st
from langchain.schema import HumanMessage, AIMessage
import concurrent.futures as cf
from database import fetch_chunks_for_profiles, fetch_profile_summaries
from embeddings import generate_embedding
from llm import analyze_query_with_llm, generate_response_with_llm
from tools import (
    education_tool,
    skills_tool,
    work_experiences_tool,
    intellectual_property_tool,
    interest_and_activities_tool,
    learnings_tool,
    offerings_tool,
    portfolio_tool,
    preferences_tool,
    profiles_tool,
    recognition_tool
)

OPENAI_API_KEY = st.secrets["openai"]["api_key"]

# ✅ Load CSV Data
CSV_FILE = r"updated_with_embeddings.csv"
df = pd.read_csv(CSV_FILE, quotechar='"', skipinitialspace=True)


# ✅ Streamlit App Configuration
st.set_page_config(page_title="RiseON Chatbot", layout="wide")
st.image(r'riseon.png', output_format='PNG', width=250)
st.title("RiseON Resume Screening")

# Initialize session states
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hi! How can I assist you?")]

# Initialize matching_profiles before accessing it
if "matching_profiles" not in st.session_state:
    st.session_state.matching_profiles = []


if "requires_retrieval" not in st.session_state:
    st.session_state.requires_retrieval = True

if "requires_response_generation" not in st.session_state:
    st.session_state.requires_response_generation = True

# Initialize profile_ids with a default list if not already set
if "profile_ids" not in st.session_state:
    st.session_state.profile_ids = df["profileid"].unique().tolist()
    # print(f"Total profiles : {len(st.session_state.profile_ids)} and {st.session_state.profile_ids}")


# Function to generate the HTML file dynamically using index.html template
def generate_html():
    # Read the static HTML template
    with open(r"C:\Users\asus\OneDrive\Desktop\streamlit\ResumeChat\index.html", "r", encoding="utf-8") as file:
        html_template = file.read()

    # Generate the table rows dynamically
    table_rows = ""
    for idx, row in df.iterrows():
        summary_id = f"summary-{idx}"
        btn_id = f"btn-summary-{idx}"

        table_rows += f"""
        <tr>
            <td>{idx + 1}</td>  <!-- SL. No. -->
            <td>{row['profileid']}</td>
            <td>{row['name']}</td>
            <td>
                <span id="{summary_id}" class="summary">{row['summary']}</span>
                <br>
                <span id="{btn_id}" class="read-more" onclick="toggleSummary('{summary_id}')">Read More</span>
            </td>
            <td>{row['total_years_workex']}</td>
            <td><a href="{row['profile_url']}" class="profile-link" target="_blank">View Profile</a></td>
        </tr>
        """

    # Replace the placeholder in the HTML template with the actual table rows
    html_content = html_template.replace("{{ table_rows }}", table_rows)

    # Save the final rendered HTML file
    html_file_path = "profile_data.html"
    with open(html_file_path, "w", encoding="utf-8") as file:
        file.write(html_content)

    return html_file_path


github_url = "https://sai-indupuri.github.io/ResumeChat/profile_data.html"

# Creating a button with an icon that looks like a native Streamlit button
st.sidebar.markdown(
    f"""
    <style>
    .open-button {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 6px 10px;
        font-size: 14px;
        font-weight: 500;
        background-color: #f0f0f5;
        color: black;
        border-radius: 5px;
        border: 1px solid #ccc;
        text-decoration: none;
        transition: 0.2s ease-in-out;
    }}

    .open-button:hover {{
        background-color: #e0e0e5;
        color: black;
    }}

    .open-button img {{
        margin-right: 8px;
    }}
    </style>

    <a href="{github_url}" target="_blank" class="open-button">
    </a>
    """,
    unsafe_allow_html=True
)



# Sidebar controls
with st.sidebar:
    st.button("Clear Chat", on_click=lambda: st.session_state.clear())
    st.markdown("### Example Queries")
    st.write("Here are some examples you can try:")
    st.markdown("""
    - "Share the candidates who are freshers or have done internships."
    - "Share the details of Sourabh Purwar and Amit Kumar."
    - "Share the candidates who can be a good fit for web development roles."
    - "Who has an MBA and AI skills?"
    """)


    # Add Company Information
    st.markdown("### About Us")
    st.write("Learn more about our products and services:")
    
    # Links to HappyPeopleAI and RiseON
    st.markdown(
        """
        - [HappyPeopleAI](https://happypeopleai.com/)
        - [RiseON](https://riseon.happypeopleai.com/)
        """
    )
# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)


def analyze_and_set_flags(user_query):
    """
    Analyze the user query to determine tools and limit.
    """
    last_message = st.session_state.chat_history[-1:]
    analysis = analyze_query_with_llm(user_query, last_message)
    tools = analysis["tools"]
    limit = int(analysis.get("limit", 10))
    return tools, limit


def prepare_context_for_llm(chat_history, max_context_messages=1):
    """
    Prepare a limited chat context for the LLM.
    """
    return chat_history[-max_context_messages:]


# User input handling
user_query = st.chat_input("Type your message...")
if user_query:
    with st.chat_message("Human"):
        st.write(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))

    # Analyze query and set execution flags
    tools, limit = analyze_and_set_flags(user_query)
    st.session_state.requires_retrieval = bool(tools)

    # Step 1: Retrieve profiles if needed
    matching_profiles = []
    profile_ids = st.session_state.profile_ids  # Get the profile IDs explicitly
    if st.session_state.requires_retrieval and tools and profile_ids:
        with st.chat_message("AI"):
            with st.spinner("Fetching matching profiles..."):
                query_embedding = generate_embedding(user_query)
                all_results = []

                # Explicitly pass profile_ids to the thread function
                def score_tool(tool, profile_ids):
                    return tool_functions[tool](profile_ids, [query_embedding])

                # Retrieve data using parallel execution
                tool_functions = {
                    "educations": education_tool,
                    "Abilities": skills_tool,
                    "workexpirences": work_experiences_tool,
                    "Intellectual_Property": intellectual_property_tool,
                    "Interest_and_Activities": interest_and_activities_tool,
                    "learnings": learnings_tool,
                    "offerings": offerings_tool,
                    "portfolio": portfolio_tool,
                    "preferences": preferences_tool,
                    "profiles": profiles_tool,
                    "Recognition": recognition_tool
                }

                with cf.ThreadPoolExecutor(max_workers=5) as executor:
                    future_to_tool = {executor.submit(score_tool, tool, profile_ids): tool for tool in tools}
                    for future in cf.as_completed(future_to_tool):
                        tool = future_to_tool[future]
                        try:
                            tool_results = future.result()
                            all_results.extend(tool_results)
                        except Exception as e:
                            print(f"Error processing tool {tool}: {e}")

                # Aggregate results
                profile_scores = {}
                for result in all_results:
                    profile_id = result["profile_id"]
                    similarity = result["similarity"]
                    if profile_id in profile_scores:
                        profile_scores[profile_id].append(similarity)
                    else:
                        profile_scores[profile_id] = [similarity]

                matching_profiles = sorted(
                    [{"profile_id": pid, "average_similarity": sum(scores) / len(scores)}
                     for pid, scores in profile_scores.items()],
                    key=lambda x: x["average_similarity"],
                    reverse=True
                )[:limit]
                st.session_state.matching_profiles = matching_profiles
                # print(f"Matching profiles:{matching_profiles}")


    if st.session_state.matching_profiles:
        profile_ids = [profile["profile_id"] for profile in st.session_state.matching_profiles]
        profiles_data = fetch_profile_summaries(profile_ids)
    else:
        profiles_data = []
        
            # Step 2: Generate response
    with st.chat_message("AI"):
        with st.spinner("Generating response..."):
            recent_history = prepare_context_for_llm(st.session_state.chat_history, max_context_messages=2)
            # profiles_data = [
            #     {chunk["chunk_type"]: json.loads(chunk["data"])
            #     for chunk in fetch_profile_summaries([profile["profile_id"]])}  # Removed .data
            #     for profile in st.session_state.matching_profiles
            # ] if st.session_state.matching_profiles else []
            # print(profiles_data)
            response_stream = generate_response_with_llm(
                user_query, profile_details=profiles_data, chat_history=recent_history
            )
        streamed_response = st.write_stream(response_stream)
        st.session_state.chat_history.append(AIMessage(content=streamed_response))

import json
import pandas as pd
import streamlit as st
from langchain.schema import HumanMessage, AIMessage
import concurrent.futures as cf
from database import fetch_chunks_by_profile
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

# Streamlit app configuration
st.set_page_config(page_title="RiseON Chatbot", layout="wide")
st.title("RiseON Resume Screening Chatbot")

# Initialize session states
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hi! How can I assist you?")]

if "matching_profiles" not in st.session_state:
    st.session_state.matching_profiles = []

if "requires_retrieval" not in st.session_state:
    st.session_state.requires_retrieval = True

if "requires_response_generation" not in st.session_state:
    st.session_state.requires_response_generation = True

# Sidebar controls
with st.sidebar:
    st.button("Clear Chat", on_click=lambda: st.session_state.clear())
    st.write("Chatbot for querying and screening resumes based on your requirements.")

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)



# Function to extract and print profile IDs and names
def print_matching_profiles(matching_profiles):
    """
    Print the matching profile IDs along with their first and last names.

    Args:
        matching_profiles (list): List of matching profile dictionaries with profile IDs and other details.
    """
    # print("\nMatching Profiles:")
    for profile in matching_profiles:
        profile_id = profile["profile_id"]
        chunks = fetch_chunks_by_profile(profile_id).data
        for chunk in chunks:
            if chunk.get("chunk_type") == "profiles":
                try:
                    # Safely parse JSON data
                    profile_data = json.loads(chunk["data"])
                    # Access nested data dictionary safely
                    data = profile_data.get("data", {})
                    first_name = data.get("first_name", "N/A")
                    last_name = data.get("last_name", "N/A")
                    print(f"Profile ID: {profile_id}, Name: {first_name} {last_name}")
                except (json.JSONDecodeError, AttributeError) as e:
                    print(f"Error parsing profile data for Profile ID {profile_id}: {e}")



# Functions
def analyze_and_set_flags(user_query):
    """
    Analyze the user query to determine tools and limit.
    """
    # Pass only the last message from the chat history
    last_message = st.session_state.chat_history[-1:]
    analysis = analyze_query_with_llm(user_query, last_message)
    tools = analysis["tools"]
    limit = int(analysis.get("limit", 5))
    return tools, limit


def prepare_context_for_llm(chat_history, max_context_messages=2):
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
    if st.session_state.requires_retrieval and tools:
        with st.chat_message("AI"):
            with st.spinner("Fetching matching profiles..."):
                query_embedding = generate_embedding(user_query)
                all_results = []

                def score_tool(tool):
                    return tool_functions[tool]([query_embedding])

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
                    future_to_tool = {executor.submit(score_tool, tool): tool for tool in tools}
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
                print("matching profiles:",matching_profiles)
                st.session_state.matching_profiles = matching_profiles
                print_matching_profiles(matching_profiles)


    # Step 2: Generate response
    with st.chat_message("AI"):
        with st.spinner("Generating response..."):
            # Prepare context for the LLM
            recent_history = prepare_context_for_llm(st.session_state.chat_history, max_context_messages=2)
            profiles_data = [
                {chunk["chunk_type"]: json.loads(chunk["data"]) 
                 for chunk in fetch_chunks_by_profile(profile["profile_id"]).data}
                for profile in st.session_state.matching_profiles
            ] if st.session_state.matching_profiles else []

            # Generate response using the LLM
            response_stream = generate_response_with_llm(
                user_query, profile_details=profiles_data, chat_history=recent_history
            )
        streamed_response = st.write_stream(response_stream)

            # Append AI response to chat history
        st.session_state.chat_history.append(AIMessage(content=streamed_response))


        
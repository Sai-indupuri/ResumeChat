import os
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.exceptions import OutputParserException
import streamlit as st
OPENAI_API_KEY = st.secrets["openai"]["api_key"]

# ✅ Function: Extract Query Metadata (LLM Classification)
def extract_query_metadata(user_query):

        # ✅ Define JSON Schema for Metadata Extraction
    response_schemas = [
        ResponseSchema(name="query_type", type="string", description="Type of query (semantic, metadata, hybrid, ranking)"),
        ResponseSchema(name="ranking_type", type="string", description="Ranking type (asc for least experience, desc for most experience) or null", default=None),
        ResponseSchema(name="job_role", type="string", description="Extracted job role from the query (e.g., 'Backend Developer', 'Data Scientist')", default=None),
        ResponseSchema(name="min_experience", type="integer", description="Minimum years of experience required", default=None),
        ResponseSchema(name="max_experience", type="integer", description="Maximum years of experience allowed", default=None),
        ResponseSchema(name="skills", type="array", description="List of extracted skills from the query (e.g., ['AWS', 'Python'])", default=[]),
    ]

    # ✅ Create Structured Output Parser
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    # ✅ Correctly Get Format Instructions
    format_instructions = output_parser.get_format_instructions()


    prompt_template = PromptTemplate(
        template="""
          You are an AI-powered recruitment assistant. 
    Your job is to analyze the user's query and extract structured search parameters.

    **Rules for Classification:**
    1. **Detect Query Type:**
       - If the query asks for **experience filtering** (e.g., "candidates with 5+ years experience"), set `"query_type": "metadata"`.
       - If both experience and skills are required (e.g., "AI engineers with 5+ years"), set `"query_type": "hybrid"`.
       - If the query asks for **most or least experienced candidates**, or implies ranking (e.g., "senior engineers", "junior ML engineers"), set `"query_type": "ranking"`.
       - If the query is **skills-based** or **education-related** (e.g., "Find AI engineers with NLP experience", "Who completed PhD"), set `"query_type": "semantic"`.
       - If the query mentions **"freshers" or "internships"**, set `"query_type": "metadata"` and `"min_experience": 0, "max_experience": 2, "ranking_type": "asc"`.

    2. **Extract Experience Filters:**
       - If "freshers" is mentioned, set `"min_experience": 0, "max_experience": 2, "ranking_type": "asc", "query_type":"ranking"`.
       - If the query specifies a **minimum** experience (e.g., "5+ years"), extract `"min_experience": 5` and `"max_experience": null`.
       - If the query specifies an **experience range** (e.g., "between 3 and 8 years"), extract `"min_experience": 3, "max_experience": 8`.
       - If the query asks for **senior-level roles**, assume `"min_experience": 5, "ranking_type": "desc"` (unless specified otherwise).
       - If the query asks for **entry-level/junior roles**, assume `"max_experience": 2, "ranking_type": "asc"`.

    3. **Handle Ranking Queries:**
       - If the query asks for **most experienced candidates**, set `"ranking_type": "desc"`.
       - If the query asks for **least experienced candidates**, junior roles, or freshers, set `"ranking_type": "asc"`.

    4. **Extract Skills & Job Roles:**
       - Identify specific **job roles** (e.g., "AI Engineer", "Software Developer", "Data Scientist").
       - Extract **skills** (e.g., "Python", "Machine Learning", "NLP", "TensorFlow").


        **Return JSON in this exact format:**
        {format_instructions}

        **User Query:** "{user_query}"
        """,
        input_variables=["user_query"],  # ✅ Ensure `user_query` is an expected variable
        partial_variables={"format_instructions": format_instructions},  # ✅ Pass format_instructions properly
    )

    # ✅ Invoke LLM
    model = ChatOpenAI(api_key=OPENAI_API_KEY,model="gpt-4o-mini", temperature=0)
    
    try:
        response = model.invoke([HumanMessage(content=prompt_template.format(user_query=user_query))])

        # ✅ Parse Response with StructuredOutputParser
        parsed_response = output_parser.parse(response.content)
        return parsed_response
    
    except OutputParserException as e:
        print(f"⚠️ Failed to parse JSON response: {e}. Falling back to default values.")
        return {
            "query_type": "semantic",
            "ranking_type": None,
            "job_role": None,
            "min_experience": None,
            "max_experience": None,
            "skills": []
        }
    



def generate_answer(user_query, profiles):
    llm = ChatOpenAI(api_key=OPENAI_API_KEY,model="gpt-4o-mini", temperature=0)

    profile_descriptions = "\n\n".join([
        f"Name: {p['name']}\nExperience: {p['total_years_workex']} years\nURL: {p['profile_url']}\nSummary: {p['summary']}"
        for p in profiles
    ])
    
    prompt = f"""
    You are a recruitment QnA assistant. Analyze the below user query and available data to answer. 
    User Query: "{user_query}"

    Candidate Profiles:
    {profile_descriptions}

    ### TASK
    1. If the query is conversational (e.g., "Hi," "How are you?"), respond in a friendly and professional manner.
    2. If profiles are provided, generate a detailed response referencing the relevant information in the profiles:
       - Include names, profile urls and purely answer to query.
    3. If no profiles are available, ask the user for more information or criteria to refine the search.
    4. Ensure the response is clear, concise, and professional.
    """
    
    stream = llm.stream([HumanMessage(content=prompt)])
    return stream

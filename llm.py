import os
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.exceptions import OutputParserException

OPENAI_API_KEY = st.secrets["openai"]["api_key"]

def analyze_query_with_llm(query, chat_history):
    """
    Analyze a user query to determine the relevant tools, profile limit, or if the query is conversational.

    Args:
        query (str): The user query.
        chat_history (list): A list of past interactions (messages) for context.

    Returns:
        dict: A dictionary containing:
              - tools: A list of tools to invoke based on the query.
              - limit: The maximum number of profiles to return (default: 5).
              - conversational: A boolean indicating if the query is purely conversational.
    """

    # Define the response schema for the LLM output
    response_schemas = [
        ResponseSchema(
            name="tools",
            description="A list of tools to invoke based on the query. Example: ['educations', 'Abilities']"
        ),
        ResponseSchema(
            name="limit",
            description="The maximum number of profiles to return. Example: 5"
        ),
        ResponseSchema(
            name="conversational",
            description="A boolean value indicating if the query is purely conversational. Example: true"
        ),
    ]

    # Create the structured output parser
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    # Include chat history in the prompt
    formatted_history = "\n".join(
        f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
        for msg in chat_history
    )

    # Define the prompt template
    PROMPT_TEMPLATE = """
    You are an intelligent assistant specializing in analyzing user queries to recommend tools, define profile limits, or engage in conversations.

    ### CHAT HISTORY
    Below is the history of the conversation so far:
    {formatted_history}

    ### TOOLS
    Below is a list of tools for retrieving specific data. Match the query with the most relevant tools:
    - **educations**: Use when the query mentions degrees, universities, certifications, or education background. Example: "Who has a degree in Computer Science?"
    - **Abilities**: Use for queries about technical skills (e.g., Python, Java), interpersonal skills (e.g., leadership, communication), or tools (e.g., Docker, Kubernetes).
    - **Intellectual_Property**: Use when the query includes terms like patents, research papers, publications, or thought leadership.
    - **Interest_and_Activities**: Use for queries about hobbies, volunteering, or supported causes.
    - **learnings**: Use for queries mentioning certifications, courses, or specific learning programs (e.g., "Who has AWS certification?").
    - **offerings**: Use when the query asks for services or products provided by candidates (e.g., "Who offers AI consulting?").
    - **portfolio**: Use when the query mentions projects, case studies, or articles authored by candidates.
    - **preferences**: Use when the query refers to location, salary expectations, job roles, or work preferences (e.g., "Who prefers remote work?").
    - **profiles**: Always use if the query explicitly mentions a candidate's name, email, or professional summary. Example: "Tell me about John Doe."
    - **Recognition**: Use for queries about awards, achievements, or recognitions (e.g., "Who won an award for AI development?").
    - **workexpirences**: Use for queries about job titles, career history, companies, or responsibilities. Example: "Who has worked at Google?"

    ### TASK
    Based on the query:
    1. **Determine Tools**: Identify the tools relevant for retrieving candidate data. If the query explicitly mentions a candidate's name, email, or other personal details, use the "profiles" tool only.
    2. **Set Profile Limit**: Determine the number of profiles to return. If not mentioned in the query, default to 5.
    3. **Determine Conversational Status**: If the query is purely conversational (e.g., "Hi," "How are you?"), set "conversational" to true and skip tools.
    4. Try to generate response n the context of query.
    QUERY:
    {query}

    FORMAT INSTRUCTIONS:
    {format_instructions}
    """

    # Initialize the ChatOpenAI instance
    llm = ChatOpenAI(
        temperature=0.5,
        model="gpt-4o-mini",
        openai_api_key=OPENAI_API_KEY
    )
    
    # Create the prompt with the user's query and history
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["formatted_history", "query", "format_instructions"]
    ).format(
        formatted_history=formatted_history,
        query=query,
        format_instructions=format_instructions
    )
    
    # Send the prompt to the LLM and parse the response
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        print(f"LLM Raw Response: {response.content}")  # Debugging output
        parsed_response = output_parser.parse(response.content)
        return parsed_response
    except OutputParserException as e:
        print(f"Error parsing LLM output: {e}")
        return {"tools": [], "limit": 5, "conversational": False, "error": True}  # Default fallback


def generate_response_with_llm(query, profile_details, chat_history, refinement_context=None):
    """
    Generate a response using the provided query, profile details, chat history, and refinement context.

    Args:
        query (str): The user query.
        profile_details (list): A list of profiles to consider for the response.
        chat_history (list): A list of past interactions (messages) for context.
        refinement_context (str): Optional string containing the refinement context.

    Returns:
        str: The generated response.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0.7)

    # Format the chat history for the LLM
    formatted_history = "\n".join(
        f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
        for msg in chat_history
    )

    # Construct the prompt
    llm_prompt = f"""
    You are a professional Talent Acquisition Assistant specializing in conversational interactions and resume screening.
    Use the query, conversation history, and profile details below to craft the best response.

    ### CHAT HISTORY
    Below is the history of the conversation so far:
    {formatted_history}

    ### QUERY
    "{query}"

    ### PROFILE DETAILS
    Here are the top matching profiles retrieved for the query:
    {profile_details if profile_details else "No profiles available."}

    ### TASK
    1. If the query is conversational (e.g., "Hi," "How are you?"), respond in a friendly and professional manner.
    2. If profiles are provided, generate a detailed response referencing the relevant information in the profiles:
       - Include names, skills, or experiences directly relevant to the query.
    3. If no profiles are available, ask the user for more information or criteria to refine the search.
    4. Ensure the response is clear, concise, and professional.
    """

    # Stream the response
    stream = llm.stream([HumanMessage(content=llm_prompt)])
    return stream

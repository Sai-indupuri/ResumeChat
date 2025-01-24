import numpy as np
from database import fetch_chunks_by_type
from embeddings import cosine_similarity
import json



def generic_tool(chunk_type, query_embeddings):
    """
    Perform similarity search for multiple query embeddings.
    """
    chunks = fetch_chunks_by_type(chunk_type)
    results = []

    for chunk in chunks.data:
        chunk_embedding = np.array(json.loads(chunk["embedding"]))
        similarity_scores = [
            cosine_similarity(query_embedding, chunk_embedding) for query_embedding in query_embeddings
        ]
        average_similarity = sum(similarity_scores) / len(similarity_scores)
        results.append({"profile_id": chunk["profile_id"], "similarity": average_similarity})

    return results



def education_tool(query_embedding):
    return generic_tool("educations", query_embedding)

def skills_tool(query_embedding):
    return generic_tool("Abilities", query_embedding)

def intellectual_property_tool(query_embedding):
    return generic_tool("Intellectual_Property", query_embedding)

def interest_and_activities_tool(query_embedding):
    return generic_tool("Interest_and_Activities", query_embedding)

def learnings_tool(query_embedding):
    return generic_tool("learnings", query_embedding)

def offerings_tool(query_embedding):
    return generic_tool("offerings", query_embedding)

def portfolio_tool(query_embedding):
    return generic_tool("portfolio", query_embedding)

def preferences_tool(query_embedding):
    return generic_tool("preferences", query_embedding)

def profiles_tool(query_embedding):
    return generic_tool("profiles", query_embedding)

def recognition_tool(query_embedding):
    return generic_tool("Recognition", query_embedding)

def work_experiences_tool(query_embedding):
    return generic_tool("workexpirences", query_embedding)
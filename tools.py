import numpy as np
from database import fetch_chunks_for_profiles
from embeddings import cosine_similarity
import json


def generic_tool(chunk_type, profile_ids, query_embeddings):
    """
    Perform similarity search for multiple query embeddings restricted to specific profile IDs.
    """
    # Fetch chunks for the provided profile IDs and chunk type
    chunks = fetch_chunks_for_profiles(profile_ids, chunk_type)
    results = []

    for chunk in chunks:
        chunk_embedding = np.array(json.loads(chunk["embedding"]))
        similarity_scores = [
            cosine_similarity(query_embedding, chunk_embedding) for query_embedding in query_embeddings
        ]
        average_similarity = sum(similarity_scores) / len(similarity_scores)
        results.append({"profile_id": chunk["profile_id"], "similarity": average_similarity})

    return results


# Tools now accept both profile IDs and query embeddings
def education_tool(profile_ids, query_embeddings):
    return generic_tool("educations", profile_ids, query_embeddings)

def skills_tool(profile_ids, query_embeddings):
    return generic_tool("Abilities", profile_ids, query_embeddings)

def intellectual_property_tool(profile_ids, query_embeddings):
    return generic_tool("Intellectual_Property", profile_ids, query_embeddings)

def interest_and_activities_tool(profile_ids, query_embeddings):
    return generic_tool("Interest_and_Activities", profile_ids, query_embeddings)

def learnings_tool(profile_ids, query_embeddings):
    return generic_tool("learnings", profile_ids, query_embeddings)

def offerings_tool(profile_ids, query_embeddings):
    return generic_tool("offerings", profile_ids, query_embeddings)

def portfolio_tool(profile_ids, query_embeddings):
    return generic_tool("portfolio", profile_ids, query_embeddings)

def preferences_tool(profile_ids, query_embeddings):
    return generic_tool("preferences", profile_ids, query_embeddings)

def profiles_tool(profile_ids, query_embeddings):
    return generic_tool("profiles", profile_ids, query_embeddings)

def recognition_tool(profile_ids, query_embeddings):
    return generic_tool("Recognition", profile_ids, query_embeddings)

def work_experiences_tool(profile_ids, query_embeddings):
    return generic_tool("workexpirences", profile_ids, query_embeddings)

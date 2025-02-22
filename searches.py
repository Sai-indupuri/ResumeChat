import re
from langchain_openai.embeddings import OpenAIEmbeddings
from supabase import create_client, Client
import streamlit as st

SUPABASE_URL = st.secrets["supabase"]["url"]
OPENAI_API_KEY = st.secrets["openai"]["api_key"]
SUPABASE_KEY = st.secrets["supabase"]["key"]



supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)


def get_embedding(text):
    """
    Generate an embedding for the given text using OpenAI's API.
    """
    try:
        return embeddings.embed_documents([text])[0]
    except Exception as e:
        raise ValueError(f"Error generating embedding: {e}")



# ✅ Function: Parse Experience
def parse_experience(text):
    if not text or not isinstance(text, str):
        return 0
    years = sum(int(x) for x in re.findall(r'(\d+)\s*years?', text))
    months = sum(int(x) for x in re.findall(r'(\d+)\s*months?', text))
    return years + (months / 12)




def hybrid_search(user_query, min_experience=None, max_experience=None, job_role=None, skills=[], top_k=10):
    """
    Hybrid Search: 
    1. Uses vector search for initial candidate retrieval.
    2. Applies metadata filtering (experience, job roles, skills).
    3. Ranks profiles using a weighted scoring system.
    """

    # ✅ Step 1: Perform Initial Vector Search
    query_embedding = embeddings.embed_documents([user_query])[0]

    response = supabase.rpc("match_profiles", {
        "query_embedding": query_embedding,
        "top_k": top_k * 3  # Retrieve extra candidates for better filtering
    }).execute()

    if not response.data:
        print("🔴 No relevant profiles found from vector search.")
        return []

    profiles = response.data  # List of matching candidates

    # ✅ Step 2: Convert Experience to Numeric & Apply Metadata Filtering
    refined_profiles = []
    
    for profile in profiles:
        profile["experience_numeric"] = parse_experience(profile["total_years_workex"])

        # 🔹 Experience Filtering
        if min_experience is not None and profile["experience_numeric"] < min_experience:
            continue
        if max_experience is not None and profile["experience_numeric"] > max_experience:
            continue

        # 🔹 Job Role Similarity (Fuzzy Matching)
        job_match_score = 0
        if job_role:
            if job_role.lower() in profile["summary"].lower():
                job_match_score = 1
            elif job_role.lower() in profile["name"].lower():
                job_match_score = 0.5  # Partial match in candidate name

        # 🔹 Skill Matching (Partial Match Support)
        skill_match_count = 0
        if skills:
            skill_match_count = sum(1 for s in skills if s.lower() in profile["summary"].lower())

        # ✅ Store Profile with Additional Metadata
        profile["job_match_score"] = job_match_score
        profile["skill_match_count"] = skill_match_count
        refined_profiles.append(profile)

    # ✅ Step 3: Weighted Scoring & Re-ranking
    for profile in refined_profiles:
        # Assign weights to each factor (adjust based on importance)
        vector_similarity = profile.get("similarity", 0)  # Assume vector similarity exists
        experience_score = profile["experience_numeric"] / 10  # Normalize experience
        job_match_score = profile["job_match_score"] * 1.5  # Give job role match higher weight
        skill_score = profile["skill_match_count"] * 1.2  # Boost candidates with more skill matches

        # Final score (tune weights as needed)
        profile["final_score"] = (
            (0.6 * vector_similarity) + 
            (0.2 * experience_score) + 
            (0.15 * job_match_score) + 
            (0.15 * skill_score)
        )

    # ✅ Step 4: Sort & Return Top Candidates
    refined_profiles.sort(key=lambda p: p["final_score"], reverse=True)
    return refined_profiles[:top_k]  # ✅ Return top candidates after filtering & ranking





def ranking_search(ranking_type="desc", min_experience=None, max_experience=None, job_role=None, skills=[], top_k=10):
    """Dynamically ranks candidates based on query type, ensuring relevance first before ranking."""
    
    # ✅ Step 1: Fetch all profiles
    response = supabase.table("ResumeTestHR") \
        .select("profileid, name, summary, total_years_workex, profile_url") \
        .execute()
    
    if not response.data:
        return []

    profiles = response.data

    # ✅ Step 2: Convert experience text to numeric values
    for profile in profiles:
        profile["experience_numeric"] = parse_experience(profile["total_years_workex"])

    # ✅ Step 3: Decide whether to apply semantic search before ranking
    is_ranking_with_filters = job_role or skills  # Ranking query + job/skills = Do semantic first

    if is_ranking_with_filters:
        # 🔹 Ensure query_text is a valid **string**
        query_text = str(job_role) if job_role else " ".join(skills)
        query_text = query_text.strip() if isinstance(query_text, str) else ""  # Ensure string format
        
        if query_text:  # Only run embedding if query text is valid
            job_role_embedding = get_embedding(query_text)  # Convert to embedding
            job_role_similarities = []

            for profile in profiles:
                profile_embedding = get_embedding(profile["summary"])  # Get profile summary embedding
                similarity_score = sum(a * b for a, b in zip(profile_embedding, job_role_embedding))
                job_role_similarities.append((profile, similarity_score))

            # ✅ Sort by relevance to job role (highest similarity first)
            sorted_by_similarity = sorted(job_role_similarities, key=lambda x: x[1], reverse=True)
            relevant_profiles = [p[0] for p in sorted_by_similarity[:top_k * 3]]  # Keep only top relevant candidates
        else:
            relevant_profiles = profiles  # No valid text for embedding, fallback to normal ranking

    else:
        # 🔹 Apply **Ranking First (No Semantic Matching)**
        relevant_profiles = profiles  # Use all profiles

    if not relevant_profiles:  
        print("⚠️ No relevant candidates found for job role. Falling back to general ranking.")
        relevant_profiles = profiles  # Use all profiles as fallback
    
    # ✅ Step 4: Now Sort Relevant Profiles by Experience
    reverse_sort = ranking_type == "desc"  # Most experienced first
    sorted_profiles = sorted(relevant_profiles, key=lambda p: p["experience_numeric"], reverse=reverse_sort)

    return sorted_profiles[:top_k]  # ✅ Return top relevant and experienced candidates


def metadata_search(min_experience=None, max_experience=None, job_role=None, top_k=15):
    """Filters candidates based on experience and job role, with semantic relevance ranking."""
    
    # ✅ Step 1: Fetch all profiles
    response = supabase.table("ResumeTestHR") \
        .select("profileid, name, summary, total_years_workex, profile_url") \
        .execute()

    if not response.data:
        print("🔴 No profiles retrieved from Supabase.")
        return []

    profiles = response.data

    # ✅ Step 2: Convert experience text to numeric values
    for profile in profiles:
        profile["experience_numeric"] = parse_experience(profile["total_years_workex"])

    # ✅ Step 3: Apply Experience Filters
    filtered_profiles = [
        p for p in profiles if
        (min_experience is None or p["experience_numeric"] >= min_experience) and
        (max_experience is None or p["experience_numeric"] <= max_experience)
    ]

    # ✅ Step 4: Handle Job Role Filtering with Semantic Search
    if job_role:
        job_role_embedding = get_embedding(job_role)
        profiles_with_similarity = []

        for p in filtered_profiles:
            profile_embedding = get_embedding(p["summary"])
            similarity = sum(a * b for a, b in zip(profile_embedding, job_role_embedding))
            profiles_with_similarity.append((p, similarity))

        # Sort by relevance to job role (highest similarity first)
        filtered_profiles = [p[0] for p in sorted(profiles_with_similarity, key=lambda x: x[1], reverse=True)]

    # ✅ Step 5: Return sorted results by experience
    return sorted(filtered_profiles, key=lambda p: p["experience_numeric"], reverse=True)[:top_k]


def vector_search(user_query, top_k=10):
    query_embedding = get_embedding(user_query)
    response = supabase.rpc("match_profiles", {
        "query_embedding": query_embedding,
        "top_k": top_k
    }).execute()
    return response.data if response.data else []

def vector_search_on_subset(user_query, embeddings_data, profiles, top_k=10):
    query_embedding = get_embedding(user_query)
    similarities = [
        (profile, sum(a * b for a, b in zip(query_embedding, embedding)))
        for profile, embedding in zip(profiles, embeddings_data)
    ]
    sorted_profiles = sorted(similarities, key=lambda x: x[1], reverse=True)
    return [p[0] for p in sorted_profiles[:top_k]]
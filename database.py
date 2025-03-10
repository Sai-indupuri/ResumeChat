# database.py
from supabase import create_client
import os
import streamlit as st

# Initialize Supabase client
# SUPABASE_URL = os.getenv('SUPABASE_URL')
# SUPABASE_KEY = os.getenv('SUPABASE_KEY')
supabase = create_client(st.secrets["supabase"]["url"], st.secrets["supabase"]["key"])

import pandas as pd

def fetch_chunks_by_type(chunk_type):
    """
    Fetch chunks from the database for a specific chunk type.
    """
    try:
        response = supabase.table("vector_profile").select("*").eq("chunk_type", chunk_type).execute()
        return response
    except Exception as e:
        raise ValueError(f"Error fetching chunks for type {chunk_type}: {e}")


def fetch_chunks_for_profiles(profile_ids, chunk_type=None):
    """
    Fetch chunks for a list of profile IDs and optionally filter by chunk type.
    """
    query = supabase.table("vector_profile").select("*").in_("profile_id", profile_ids)
    if chunk_type:
        query = query.eq("chunk_type", chunk_type)
    try:
        response = query.execute()
        return response.data
    except Exception as e:
        raise ValueError(f"Error fetching chunks for profile IDs {profile_ids}: {e}")
    
# Load CSV file
CSV_FILE = r"updated_with_embeddings.csv"
df = pd.read_csv(CSV_FILE)
def fetch_profile_summaries(profile_ids):
    """
    Fetch name, summary, total_years_workex, and profile_url from the CSV for given profile IDs.
    """
    try:
        filtered_profiles = df[df["profileid"].isin(profile_ids)][
            ["profileid", "name", "summary", "total_years_workex", "profile_url"]
        ].to_dict(orient="records")
        return filtered_profiles
    except Exception as e:
        raise ValueError(f"Error fetching summaries for profile IDs {profile_ids}: {e}")

# database.py
from supabase import create_client
import os
import streamlit as st

# Initialize Supabase client
# SUPABASE_URL = os.getenv('SUPABASE_URL')
# SUPABASE_KEY = os.getenv('SUPABASE_KEY')
supabase = create_client(st.secrets["supabase"]["url"], st.secrets["supabase"]["key"])


def fetch_chunks_by_type(chunk_type):
    """
    Fetch chunks from the database for a specific chunk type.
    """
    try:
        response = supabase.table("vector_profile").select("*").eq("chunk_type", chunk_type).execute()
        return response
    except Exception as e:
        raise ValueError(f"Error fetching chunks for type {chunk_type}: {e}")


def fetch_chunks_by_profile(profile_id):
    """
    Fetch all chunks for a specific profile ID.
    """
    try:
        response = supabase.table("vector_profile").select("*").eq("profile_id", profile_id).execute()
        return response
    except Exception as e:
        raise ValueError(f"Error fetching chunks for profile ID {profile_id}: {e}")


def aggregate_profile_data(profile_ids):
    """
    Aggregate data for the given profile IDs into a structured format.

    Args:
        profile_ids (list): A list of profile IDs.

    Returns:
        dict: A dictionary where each key is a profile ID, and the value is aggregated data from all chunks.
    """
    profiles_data = {}
    try:
        for profile_id in profile_ids:
            response = fetch_chunks_by_profile(profile_id)
            chunks = response.data
            aggregated_data = {}

            # Aggregate chunks based on their types
            for chunk in chunks:
                chunk_type = chunk.get("chunk_type")
                chunk_data = chunk.get("data")
                if chunk_type and chunk_data:
                    aggregated_data[chunk_type] = chunk_data

            profiles_data[profile_id] = aggregated_data

        return profiles_data
    except Exception as e:
        raise ValueError(f"Error aggregating profile data: {e}")

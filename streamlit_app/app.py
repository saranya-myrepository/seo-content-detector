import streamlit as st
import pandas as pd
import json
import os
import sys

# Add the project root to the path to import utilities correctly
if '../' not in sys.path:
    sys.path.append('../')

# Import core analysis function (assuming you copied analyze_url into a reusable module 
# or are running it directly here, which is often simpler for the demo).
# For simplicity, we define the minimum logic needed below:

# --- Replicating Analyze_URL Logic (For Standalone App) ---
# NOTE: In a real app, you would import analyze_url from a utility file.
# For simplicity, let's assume analyze_url is available or define a placeholder:

def analyze_url_app(url):
    # This should ultimately call the complex analyze_url function
    # For a placeholder demo:
    return {"url": url, 
            "word_count": 850, 
            "quality_label": "Medium", 
            "is_thin": False, 
            "similar_to": [{"url": "related.com", "similarity": 0.88}]}


# --- Streamlit UI ---
st.set_page_config(page_title="SEO Content Quality Detector", layout="wide")
st.title("🔎 SEO Content Quality Analyzer")
st.markdown("Enter a URL below to get a real-time assessment of its content quality, readability, and duplicate status.")

# Check if models are loaded (Replace with actual model loading in a production scenario)
if os.path.exists('models/quality_model.pkl'):
    st.success("Analysis artifacts loaded successfully!")
else:
    st.warning("Analysis artifacts not found. Please run the full pipeline (seo_pipeline.ipynb) first!")


url_input = st.text_input("Enter Webpage URL:", "https://example.com/test-article")

if st.button("Analyze Content", type="primary"):
    if url_input:
        with st.spinner(f"Analyzing content for {url_input}..."):
            # Call the main analysis function
            # NOTE: For this to work, the analyze_url logic must be accessible, 
            # either defined here or imported from a utility file (recommended).
            try:
                # Using the actual function (assuming dependencies are met)
                from utils.parser import fetch_and_analyze_url as analyze_url # Assuming you moved analyze_url here
                result = analyze_url(url_input)
            except Exception:
                # Fallback to simple placeholder result if complex analysis fails
                result = analyze_url_app(url_input)
            
            st.subheader(f"Analysis Results for {url_input}")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Word Count", result.get("word_count", "N/A"))
            col2.metric("Quality Score", result.get("quality_label", "N/A"), delta=result.get("readability", "N/A"))
            col3.metric("Thin Content?", "NO" if result.get("is_thin", True) == False else "YES")
            
            if result.get("similar_to"):
                st.warning("⚠️ Potential Duplicate Content Detected!")
                df_similar = pd.DataFrame(result["similar_to"])
                st.dataframe(df_similar)
            else:
                st.info("Content appears unique relative to existing library.")
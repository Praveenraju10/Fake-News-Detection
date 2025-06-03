import streamlit as st
import pickle
import numpy as np
import os
from serpapi import GoogleSearch

# Load the trained model and vectorizer
with open("fake_news_model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

# 🔐 SerpAPI key - set via environment variable or paste below (not recommended for production)
SERPAPI_KEY = os.getenv("SERPAPI_KEY") or "2fcb593e70cd070e1058af6007e3a91c9d6875c546e2664a4808907752c1ae6d"

# Trusted news domains
TRUSTED_SITES = [
    'bbc.com', 'cnn.com', 'reuters.com', 'nytimes.com',
    'theguardian.com', 'npr.org', 'apnews.com', 'forbes.com',
    'bloomberg.com', 'cbsnews.com', 'abcnews.go.com'
]

# Streamlit UI
st.set_page_config(page_title="Fake News Detection", layout="centered")
st.title("📰 Fake News Detection App")
st.write("Enter a news article below to check if it's potentially fake using AI + real-time web verification.")

# User input
user_input = st.text_area("📝 Paste News Article (Title + Text):", height=300)

# Fixed threshold
threshold = 0.5

# Search online
def search_online(query):
    params = {
        "q": query,
        "api_key": SERPAPI_KEY,
        "engine": "google",
        "num": 5,
        "hl": "en"
    }
    search = GoogleSearch(params)
    return search.get_dict().get("organic_results", [])

# Count trusted source matches
def count_trusted_sources(results):
    count = 0
    for result in results:
        link = result.get("link", "")
        for domain in TRUSTED_SITES:
            if domain in link:
                count += 1
                break
    return count

# Detect button
if st.button("🔍 Detect Fake News"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        # Step 1: Predict with ML model
        input_vec = vectorizer.transform([user_input])
        probability = model.predict_proba(input_vec)[0][1]

        # Step 2: Search the web
        with st.spinner("🔍 Searching online..."):
            search_results = search_online(user_input[:200])
            trusted_hits = count_trusted_sources(search_results)

        # Step 3: Show scores
        st.markdown("---")
        st.subheader("🔎 Analysis Results")
        st.write(f"🤖 **AI Model Prediction Score (Fake Probability)**: `{probability:.4f}`")
        st.write(f"🌐 **Trusted News Sources Found**: `{trusted_hits}`")

        # Step 4: Final verdict
        st.markdown("### 🧠 Final Verdict")
        if probability > 0.8 and trusted_hits == 0:
            st.error("🟥 Likely FAKE: High fake score & no trusted sources.")
        elif probability < 0.2 and trusted_hits >= 2:
            st.success("🟩 Likely REAL: Low fake score & trusted confirmations.")
        elif trusted_hits >= 3:
            st.success("🟩 Probably Real: Found on multiple trusted sources.")
        elif trusted_hits == 0 and probability > 0.6:
            st.warning("🟨 Possibly FAKE: No verification & model suspects fake.")
        else:
            st.info("⚪ Inconclusive: Needs more evidence or trusted sources.")

        # Step 5: Show raw search results
        if search_results:
            st.markdown("---")
            st.subheader("🔗 Online Sources")
            for res in search_results:
                title = res.get("title", "No Title")
                link = res.get("link", "#")
                snippet = res.get("snippet", "No description")
                st.markdown(f"- **[{title}]({link})**\n    - {snippet}")
        else:
            st.info("No related results found online.")

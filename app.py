import streamlit as st
import requests
import os


st.title("Multi-Cloud RAG Assistant ☁️")

query = st.text_input("Ask your question:", placeholder="e.g. What is AWS Well-Architected Framework?")

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

if query.strip():
    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                f"{API_URL}/ask",
                json={"question": query}
            )
            response.raise_for_status()
            result = response.json()

            st.subheader("Answer")
            st.write(result.get("answer", "No answer returned"))

            st.subheader("Sources")
            for s in result.get("sources", []):
                st.write(f"- {s.get('provider')} | {s.get('title')}")

        except Exception as e:
            st.error(f"Error connecting to API: {e}")

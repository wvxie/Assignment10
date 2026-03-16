import requests
import streamlit as st

HF_ENDPOINT = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

st.set_page_config(page_title="My AI Chat", layout="wide")
st.title("My AI Chat")

# Load token from Streamlit secrets
try:
    hf_token = st.secrets["HF_TOKEN"]
except KeyError:
    hf_token = ""

if not hf_token:
    st.error(
        "Missing HF_TOKEN. Add it to Streamlit secrets (Settings -> Secrets) or .streamlit/secrets.toml."
    )
    st.stop()

headers = {"Authorization": f"Bearer {hf_token}"}
payload = {
    "model": HF_MODEL,
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 512,
}

st.write("Sending test message to the model...")

try:
    response = requests.post(HF_ENDPOINT, headers=headers, json=payload, timeout=60)
    if response.status_code != 200:
        st.error(f"HF API error {response.status_code}: {response.text[:300]}")
    else:
        data = response.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        if not content:
            content = "(No response text returned)"
        st.subheader("Model response")
        st.write(content)
except Exception as e:
    st.error(f"Request failed: {type(e).__name__}: {e}")

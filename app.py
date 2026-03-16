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


def request_completion(messages, token):
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "model": HF_MODEL,
        "messages": messages,
        "max_tokens": 512,
    }
    return requests.post(HF_ENDPOINT, headers=headers, json=payload, timeout=60)


# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render conversation history
for msg in st.session_state.messages:
    role = msg.get("role", "assistant")
    content = msg.get("content", "")
    with st.chat_message(role):
        st.write(content)

# Input bar fixed at the bottom
user_input = st.chat_input("Type your message...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            response = request_completion(st.session_state.messages, hf_token)
            if response.status_code != 200:
                placeholder.error(
                    f"HF API error {response.status_code}: {response.text[:300]}"
                )
            else:
                data = response.json()
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                if not content:
                    content = "(No response text returned)"
                placeholder.write(content)
                st.session_state.messages.append({"role": "assistant", "content": content})
        except Exception as e:
            placeholder.error(f"Request failed: {type(e).__name__}: {e}")

# Streamlit chat app using Hugging Face Inference Router
import json
from pathlib import Path

import requests
import streamlit as st

HF_ENDPOINT = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

MEMORY_PATH = Path("memory.json")
CHAT_DIR = Path("chats")


st.set_page_config(page_title="Assignment 10 Chat", page_icon="chat")
st.title("Assignment 10 Chat")

# Load token from Streamlit secrets
hf_token = st.secrets.get("HF_TOKEN")
if not hf_token:
    st.error(
        "Missing HF_TOKEN. Add it to Streamlit secrets (Settings ? Secrets) or .streamlit/secrets.toml."
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

# Load persisted chat if available
if "messages" not in st.session_state:
    messages = []
    if MEMORY_PATH.exists():
        try:
            messages = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
            if not isinstance(messages, list):
                messages = []
        except Exception:
            messages = []
    st.session_state.messages = messages

# Sidebar controls
with st.sidebar:
    st.subheader("Controls")
    if st.button("Clear chat"):
        st.session_state.messages = []
        try:
            if MEMORY_PATH.exists():
                MEMORY_PATH.write_text("[]", encoding="utf-8")
        except Exception:
            pass

# Render chat history
for msg in st.session_state.messages:
    role = msg.get("role", "assistant")
    content = msg.get("content", "")
    with st.chat_message(role):
        st.write(content)


# Chat input
prompt = st.chat_input("Ask something...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

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

    # Persist chat to memory.json
    try:
        MEMORY_PATH.write_text(
            json.dumps(st.session_state.messages, indent=2), encoding="utf-8"
        )
    except Exception:
        pass

    # Append a simple log entry in chats/ (optional)
    try:
        CHAT_DIR.mkdir(exist_ok=True)
        log_path = CHAT_DIR / "latest.txt"
        log_path.write_text(
            json.dumps(st.session_state.messages, indent=2), encoding="utf-8"
        )
    except Exception:
        pass

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import requests
import streamlit as st

HF_ENDPOINT = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
BASE_DIR = Path(__file__).resolve().parent
CHAT_DIR = BASE_DIR / "chats"

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


def now_stamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def summarize_title(text, max_words=5):
    words = text.replace("\n", " ").strip().split()
    if not words:
        return "New Chat"
    summary = " ".join(words[:max_words])
    if len(words) > max_words:
        summary += "..."
    return summary


def chat_path(chat_id):
    return CHAT_DIR / f"{chat_id}.json"


def save_chat(chat):
    CHAT_DIR.mkdir(exist_ok=True)
    data = {
        "id": chat["id"],
        "title": chat["title"],
        "timestamp": chat["timestamp"],
        "messages": chat["messages"],
        "title_set": chat.get("title_set", False),
    }
    chat_path(chat["id"]).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_chats():
    CHAT_DIR.mkdir(exist_ok=True)
    chats = []
    for path in sorted(CHAT_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        chat_id = str(data.get("id") or path.stem)
        title = data.get("title") or "New Chat"
        timestamp = data.get("timestamp") or now_stamp()
        messages = data.get("messages") if isinstance(data.get("messages"), list) else []
        title_set = bool(data.get("title_set")) if "title_set" in data else (title != "New Chat")
        chats.append(
            {
                "id": chat_id,
                "title": title,
                "timestamp": timestamp,
                "messages": messages,
                "title_set": title_set,
            }
        )
    return chats


def make_new_chat(index):
    chat = {
        "id": str(uuid4()),
        "title": f"New Chat {index}",
        "timestamp": now_stamp(),
        "messages": [],
        "title_set": False,
    }
    save_chat(chat)
    return chat


# Initialize chat list from disk
if "chats" not in st.session_state:
    st.session_state.chats = load_chats()
    if not st.session_state.chats:
        st.session_state.chats = [make_new_chat(1)]

if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = (
        st.session_state.chats[0]["id"] if st.session_state.chats else None
    )


# Sidebar UI for chat navigation
with st.sidebar:
    st.header("Chats")
    if st.button("New Chat"):
        new_chat = make_new_chat(len(st.session_state.chats) + 1)
        st.session_state.chats.append(new_chat)
        st.session_state.active_chat_id = new_chat["id"]

    list_container = st.container(height=350)
    delete_id = None
    select_id = None

    with list_container:
        if not st.session_state.chats:
            st.info("No chats yet. Click New Chat to start one.")
        else:
            for chat in st.session_state.chats:
                is_active = chat["id"] == st.session_state.active_chat_id
                label = f"{chat['title']} - {chat['timestamp']}"
                col1, col2 = st.columns([0.85, 0.15])
                with col1:
                    if st.button(
                        label,
                        key=f"select_{chat['id']}",
                        type="primary" if is_active else "secondary",
                    ):
                        select_id = chat["id"]
                with col2:
                    if st.button("✕", key=f"delete_{chat['id']}"):
                        delete_id = chat["id"]

    if delete_id:
        st.session_state.chats = [
            c for c in st.session_state.chats if c["id"] != delete_id
        ]
        try:
            chat_path(delete_id).unlink()
        except FileNotFoundError:
            pass
        if st.session_state.active_chat_id == delete_id:
            st.session_state.active_chat_id = (
                st.session_state.chats[0]["id"] if st.session_state.chats else None
            )

    if select_id:
        st.session_state.active_chat_id = select_id


# Resolve active chat
active_chat = None
for c in st.session_state.chats:
    if c["id"] == st.session_state.active_chat_id:
        active_chat = c
        break

if active_chat is None:
    st.info("No active chat. Create a new chat from the sidebar.")
    st.stop()


# Render conversation history
for msg in active_chat["messages"]:
    role = msg.get("role", "assistant")
    content = msg.get("content", "")
    with st.chat_message(role):
        st.write(content)

# Input bar fixed at the bottom
user_input = st.chat_input("Type your message...")
if user_input:
    if not active_chat.get("title_set"):
        active_chat["title"] = summarize_title(user_input)
        active_chat["title_set"] = True
        active_chat["timestamp"] = now_stamp()

    active_chat["messages"].append({"role": "user", "content": user_input})
    save_chat(active_chat)

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            response = request_completion(active_chat["messages"], hf_token)
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
                active_chat["messages"].append({"role": "assistant", "content": content})
                save_chat(active_chat)
        except Exception as e:
            placeholder.error(f"Request failed: {type(e).__name__}: {e}")
            save_chat(active_chat)

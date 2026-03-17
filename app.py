import json
from datetime import datetime
from uuid import uuid4

import requests
import streamlit as st

HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_SYSTEM_MESSAGE = {"role": "system", "content": "You are a helpful assistant."}


def chat_completion(*, hf_token: str, messages: list[dict], temperature: float, max_tokens: int) -> str:
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        resp = requests.post(HF_ROUTER_URL, headers=headers, json=payload, timeout=60)
    except requests.RequestException as exc:
        raise RuntimeError(f"Network error calling Hugging Face Router: {exc}") from exc

    content_type = resp.headers.get("content-type", "")
    if "application/json" in content_type.lower():
        try:
            data = resp.json()
        except ValueError:
            data = None
    else:
        data = None

    if not resp.ok:
        body_preview = resp.text.strip()
        if data is not None:
            body_preview = json.dumps(data, ensure_ascii=False)[:2000]
        raise RuntimeError(f"Router error {resp.status_code}: {body_preview[:2000]}")

    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected response (non-JSON): {resp.text[:2000]}")

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected response shape: {json.dumps(data, ensure_ascii=False)[:2000]}") from exc


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def format_timestamp(iso_string: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_string)
    except ValueError:
        return iso_string
    return dt.strftime("%Y-%m-%d %H:%M")


def new_chat() -> dict:
    timestamp = now_iso()
    return {
        "id": str(uuid4()),
        "title": "New chat",
        "created_at": timestamp,
        "updated_at": timestamp,
        "messages": [DEFAULT_SYSTEM_MESSAGE.copy()],
    }


def get_active_chat() -> dict | None:
    active_chat_id = st.session_state.get("active_chat_id")
    for chat in st.session_state.get("chats", []):
        if chat.get("id") == active_chat_id:
            return chat
    return None


st.set_page_config(page_title="My AI Chat", layout="wide")
st.title("My AI Chat")
st.caption(f"Using Hugging Face Inference Router • Model: `{MODEL}`")

try:
    hf_token = st.secrets["HF_TOKEN"]
except KeyError:
    hf_token = ""
except Exception:
    hf_token = ""
if not hf_token.strip():
    st.error(
        "Missing `HF_TOKEN` in `.streamlit/secrets.toml`.\n\n"
        "Add a line like:\n"
        '`HF_TOKEN = "hf_..."`\n\n'
        "Then reload the app."
    )
    st.stop()

if "chats" not in st.session_state:
    st.session_state["chats"] = []
if "active_chat_id" not in st.session_state:
    st.session_state["active_chat_id"] = None

if "messages" in st.session_state and not st.session_state["chats"]:
    migrated = new_chat()
    migrated["messages"] = st.session_state["messages"]
    migrated["title"] = "Migrated chat"
    migrated["updated_at"] = now_iso()
    st.session_state["chats"] = [migrated]
    st.session_state["active_chat_id"] = migrated["id"]
    st.session_state.pop("messages", None)

if st.session_state["active_chat_id"] is None and st.session_state["chats"]:
    most_recent = max(st.session_state["chats"], key=lambda c: c.get("updated_at", ""))
    st.session_state["active_chat_id"] = most_recent.get("id")

with st.sidebar:
    st.header("Settings")
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
    max_tokens = st.slider("Max tokens", min_value=16, max_value=2048, value=256, step=16)

    if st.button("New Chat", type="primary", use_container_width=True):
        created = new_chat()
        st.session_state["chats"].append(created)
        st.session_state["active_chat_id"] = created["id"]
        st.rerun()

    active_chat = get_active_chat()
    if st.button("Clear chat", disabled=active_chat is None, use_container_width=True):
        if active_chat is not None:
            active_chat["messages"] = [DEFAULT_SYSTEM_MESSAGE.copy()]
            active_chat["updated_at"] = now_iso()
        st.rerun()

    st.divider()
    st.subheader("Chats")

    chat_list = st.container(height=500)
    sorted_chats = sorted(
        st.session_state["chats"], key=lambda c: c.get("updated_at", ""), reverse=True
    )
    with chat_list:
        for chat in sorted_chats:
            chat_id = chat.get("id", "")
            title = chat.get("title", "Chat")
            timestamp = format_timestamp(chat.get("updated_at", chat.get("created_at", "")))
            is_active = chat_id == st.session_state.get("active_chat_id")

            cols = st.columns([0.85, 0.15], vertical_alignment="center")
            with cols[0]:
                if st.button(
                    f"{title} • {timestamp}",
                    key=f"open_{chat_id}",
                    type="primary" if is_active else "secondary",
                    use_container_width=True,
                ):
                    st.session_state["active_chat_id"] = chat_id
                    st.rerun()
            with cols[1]:
                if st.button("✕", key=f"del_{chat_id}", type="tertiary", use_container_width=True):
                    st.session_state["chats"] = [c for c in st.session_state["chats"] if c.get("id") != chat_id]
                    if st.session_state.get("active_chat_id") == chat_id:
                        if st.session_state["chats"]:
                            most_recent_remaining = max(
                                st.session_state["chats"], key=lambda c: c.get("updated_at", "")
                            )
                            st.session_state["active_chat_id"] = most_recent_remaining.get("id")
                        else:
                            st.session_state["active_chat_id"] = None
                    st.rerun()

active_chat = get_active_chat()
if active_chat is None:
    st.info("No chats yet. Click **New Chat** in the sidebar to start.")
    st.stop()

history = st.container(height=600)
with history:
    for msg in active_chat.get("messages", []):
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            continue
        with st.chat_message(role):
            st.markdown(content)

user_text = st.chat_input("Type your message…")
if user_text:
    user_text = str(user_text)
    active_chat["messages"].append({"role": "user", "content": user_text})
    active_chat["updated_at"] = now_iso()
    if active_chat.get("title") == "New chat":
        active_chat["title"] = (user_text.strip() or "New chat")[:40]

    with history:
        with st.chat_message("user"):
            st.markdown(user_text)

    with history:
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    assistant_text = chat_completion(
                        hf_token=hf_token,
                        messages=active_chat["messages"],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                except RuntimeError as exc:
                    error_text = str(exc)
                    if "Router error 401" in error_text or "Router error 403" in error_text:
                        error_text = (
                            "Authentication failed (401/403). Your `HF_TOKEN` is missing/invalid or lacks permissions."
                        )
                    elif "Router error 429" in error_text:
                        error_text = "Rate limit exceeded (429). Please wait a bit and try again."
                    st.error(error_text)
                    assistant_text = f"Error: {error_text}"
                else:
                    st.markdown(assistant_text)

    active_chat["messages"].append({"role": "assistant", "content": assistant_text})
    active_chat["updated_at"] = now_iso()

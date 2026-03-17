import json

import requests
import streamlit as st

HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL = "meta-llama/Llama-3.2-1B-Instruct"


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

with st.sidebar:
    st.header("Settings")
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
    max_tokens = st.slider("Max tokens", min_value=16, max_value=2048, value=256, step=16)
    if st.button("Clear chat"):
        st.session_state["messages"] = [{"role": "system", "content": "You are a helpful assistant."}]
        st.rerun()

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "system", "content": "You are a helpful assistant."}]

history = st.container(height=600)
with history:
    for msg in st.session_state["messages"]:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            continue
        with st.chat_message(role):
            st.markdown(content)

user_text = st.chat_input("Type your message…")
if user_text:
    st.session_state["messages"].append({"role": "user", "content": user_text})
    with history:
        with st.chat_message("user"):
            st.markdown(user_text)

    with history:
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    assistant_text = chat_completion(
                        hf_token=hf_token,
                        messages=st.session_state["messages"],
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
                    st.session_state["messages"].append({"role": "assistant", "content": f"Error: {error_text}"})
                else:
                    st.markdown(assistant_text)
                    st.session_state["messages"].append({"role": "assistant", "content": assistant_text})

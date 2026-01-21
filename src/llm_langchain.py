# src/llm_langchain.py
from langchain_openai import ChatOpenAI


def get_local_qwen_model() -> ChatOpenAI:
    """
    Wrap a local vLLM (OpenAI-compatible) endpoint as a LangChain ChatOpenAI model.

    Assumptions:
    - vLLM is running at http://12.12.12.62:8000/v1
    - Model name is 'qwen2.5-coder-7b' (adjust if needed)
    """
    llm = ChatOpenAI(
        openai_api_base="http://12.12.12.62:8000/v1",
        openai_api_key="EMPTY",  # vLLM typically ignores this, but field is required
        model="qwen2.5-coder-7b",
        temperature=0.2,
        max_tokens=512,
    )
    return llm

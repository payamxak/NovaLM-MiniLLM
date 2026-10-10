def format_chat(messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str:
    """Format role/content messages using NovaLM's explicit role token convention."""
    allowed = {"system", "user", "assistant", "tool"}
    parts: list[str] = []
    for message in messages:
        role, content = message.get("role"), message.get("content")
        if role not in allowed or not isinstance(content, str):
            raise ValueError("each message needs a supported role and string content")
        parts.append(f"<|{role}|>\n{content}\n")
    if add_generation_prompt:
        parts.append("<|assistant|>\n")
    return "".join(parts)

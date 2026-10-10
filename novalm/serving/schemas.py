from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=64, ge=1, le=1024)
    temperature: float = Field(default=0.8, ge=0, allow_inf_nan=False)
    top_p: float = Field(default=0.95, gt=0, le=1, allow_inf_nan=False)
    top_k: int = Field(default=0, ge=0)
    min_p: float = Field(default=0.0, ge=0, le=1, allow_inf_nan=False)
    repetition_penalty: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    stop: str | list[str] | None = None
    stream: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(CompletionRequest):
    prompt: str | None = None
    messages: list[ChatMessage] | None = None

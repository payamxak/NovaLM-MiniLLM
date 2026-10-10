import json
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from novalm.inference.generate import stream_generate
from novalm.serving.schemas import ChatCompletionRequest, CompletionRequest
from novalm.tokenizer.chat_template import format_chat


def create_app(model, tokenizer, model_name: str = "novalm") -> FastAPI:
    app = FastAPI(title="NovaLM API", version="0.1.0")
    app.state.model, app.state.tokenizer, app.state.model_name = model, tokenizer, model_name

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/model/info")
    def model_info():
        return {"id": app.state.model_name, "parameters": model.parameter_count,
                "context_length": model.config.max_seq_length}

    def completion(prompt: str, request, chat: bool = False):
        if len(prompt) > 100_000:
            raise HTTPException(status_code=413, detail="prompt is too long")
        ids = tokenizer.encode(prompt)
        if not ids:
            raise HTTPException(status_code=422, detail="prompt produced no tokens")
        if len(ids) + request.max_new_tokens > model.config.max_seq_length:
            raise HTTPException(status_code=422, detail="prompt plus generation exceeds model context length")
        stop_values = ([request.stop] if isinstance(request.stop, str) else request.stop) or []
        stop_sequences = [tokenizer.encode(value) for value in stop_values]
        if any(not sequence for sequence in stop_sequences):
            raise HTTPException(status_code=422, detail="stop strings must encode to at least one token")
        eos_token_id = tokenizer.processor.eos_id()

        def finish_reason(generated_ids: list[int]) -> str:
            if len(generated_ids) < request.max_new_tokens:
                return "stop"
            if generated_ids and generated_ids[-1] == eos_token_id:
                return "stop"
            if any(len(stop) <= len(generated_ids) and generated_ids[-len(stop):] == stop
                   for stop in stop_sequences if stop):
                return "stop"
            return "length"

        device = next(model.parameters()).device
        import torch
        input_ids = torch.tensor([ids], device=device)
        completion_id = f"cmpl-{uuid.uuid4().hex}"
        if request.stream:
            def events():
                generated_ids: list[int] = []
                for token in stream_generate(model, input_ids, request.max_new_tokens,
                                             eos_token_id=eos_token_id,
                                             temperature=request.temperature, top_k=request.top_k,
                                             top_p=request.top_p, min_p=request.min_p,
                                             repetition_penalty=request.repetition_penalty,
                                             stop_sequences=stop_sequences):
                    generated_ids.extend(token[0].tolist())
                    piece = tokenizer.decode(token[0].tolist())
                    if chat:
                        choice = {"index": 0, "delta": {"content": piece}, "finish_reason": None}
                        object_type = "chat.completion.chunk"
                    else:
                        choice = {"index": 0, "text": piece, "finish_reason": None}
                        object_type = "text_completion.chunk"
                    chunk = {"id": completion_id, "object": object_type, "choices": [choice]}
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                reason = finish_reason(generated_ids)
                if chat:
                    final_choice = {"index": 0, "delta": {}, "finish_reason": reason}
                    final_object = "chat.completion.chunk"
                else:
                    final_choice = {"index": 0, "text": "", "finish_reason": reason}
                    final_object = "text_completion.chunk"
                yield f"data: {json.dumps({'id': completion_id, 'object': final_object, 'choices': [final_choice]})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(events(), media_type="text/event-stream")
        started = time.perf_counter()
        generated = list(stream_generate(model, input_ids, request.max_new_tokens,
                                         eos_token_id=eos_token_id,
                                         temperature=request.temperature, top_k=request.top_k,
                                         top_p=request.top_p, min_p=request.min_p,
                                         repetition_penalty=request.repetition_penalty,
                                         stop_sequences=stop_sequences))
        ids_out = torch.cat([input_ids, *generated], dim=1) if generated else input_ids
        text = tokenizer.decode(ids_out[0, len(ids):].tolist())
        generated_ids = [int(token[0, 0]) for token in generated]
        reason = finish_reason(generated_ids)
        choice = ({"index": 0, "message": {"role": "assistant", "content": text},
                   "finish_reason": reason}
                  if chat else {"index": 0, "text": text,
                                "finish_reason": reason})
        return {"id": completion_id, "object": "chat.completion" if chat else "text_completion",
                "created": int(time.time()), "model": model_name, "choices": [choice],
                "usage": {"prompt_tokens": len(ids), "completion_tokens": len(generated),
                          "total_tokens": len(ids) + len(generated),
                          "elapsed_seconds": time.perf_counter() - started}}

    @app.post("/v1/completions")
    def completions(request: CompletionRequest):
        return completion(request.prompt, request)

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest):
        if not request.messages:
            raise HTTPException(status_code=422, detail="messages are required")
        try:
            prompt = format_chat([m.model_dump() for m in request.messages],
                                 add_generation_prompt=True)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return completion(prompt, request, chat=True)

    return app


import os
import time
import secrets
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from chat_app import Session, Options, get_assistant

app = FastAPI(title="Ammo Card API", version="1.0.0")

# أضف نطاق موقعك في إعداد CORS_ORIGINS على الاستضافة
origins = [
    x.strip()
    for x in os.getenv("CORS_ORIGINS", "").split(",")
    if x.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

API_KEY = os.getenv("EIM_API_KEY", "")
MAX_PROMPT_CHARS = 12000


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "ammo-card"
    messages: list[Message] = Field(min_length=1, max_length=30)
    temperature: float = Field(default=0.4, ge=0, le=1.2)
    max_tokens: int = Field(default=512, ge=1, le=1024)


def generate_reply(request: ChatRequest) -> str:
    # نستخدم واجهات المشروع الفعلية، وليس endpoint مفترضاً
    messages = [
        {"role": m.role, "content": m.content}
        for m in request.messages
    ]

    user_positions = [
        i for i, m in enumerate(messages)
        if m["role"] == "user"
    ]
    if not user_positions:
        raise ValueError("يجب أن يحتوي الطلب على رسالة user")

    last_user = user_positions[-1]
    prompt = messages[last_user]["content"]

    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError("الرسالة أطول من الحد المسموح")

    # الجلسة تحمل سجل المحادثة السابق
    session = Session()
    session.history = [
        m for m in messages[:last_user]
        if m["role"] in ("user", "assistant")
    ][-20:]

    options = Options(
        mode="chat",
        temperature=request.temperature,
        max_new_tokens=request.max_tokens,
        context_chars=6000,
    )

    # handle() يعيد أجزاء الإجابة تدريجياً؛ نأخذ الإجابة النهائية
    answer = ""
    for snapshot in get_assistant().handle(
        session, prompt, [], options
    ):
        answer = snapshot

    if not answer:
        raise RuntimeError("لم يُرجع النموذج إجابة")

    return answer


@app.get("/health")
def health():
    return {"status": "ok", "service": "ammo-card-api"}


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatRequest,
    authorization: str | None = Header(default=None),
):
    if not API_KEY:
        raise HTTPException(
            status_code=503,
            detail="API key is not configured on the server",
        )

    supplied = ""
    if authorization and authorization.startswith("Bearer "):
        supplied = authorization[7:].strip()

    if not secrets.compare_digest(supplied, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")

    started = time.time()

    try:
        answer = await run_in_threadpool(generate_reply, request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        # لا نكشف تفاصيل الاستثناءات الداخلية للزوار
        raise HTTPException(status_code=500, detail="Model request failed")

    return {
        "id": "chatcmpl-" + uuid4().hex,
        "object": "chat.completion",
        "created": int(started),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer,
                },
                "finish_reason": "stop",
            }
        ],
    }
    
# echo.py — ChatGPT bilan javob beradigan echo-router (parallel + retry + code-aware)
# --------------------------------------------------------------
# Xususiyatlar:
#  - AsyncOpenAI bilan javoblar
#  - Semaphore (parallel so'rovlar sonini cheklaydi, lekin bir vaqtda ko'p userga xizmat qiladi)
#  - Retry + exponential backoff (429/vaqtinchalik xatolar uchun)
#  - Kontekst xotira (har user uchun oxirgi 8 xabar)
#  - Javobni chiroyli formatda yuborish: oddiy matn bo'lib yuboriladi; kod bo'lsa code block ko'rinishida
#  - /reset bilan user konteksti tozalanadi

import os
import re
import asyncio
from collections import defaultdict, deque
from textwrap import wrap
from typing import Deque, Dict, List, Optional, Tuple

from aiogram import types, Router, F
from aiogram.filters import StateFilter, Command
from aiogram.fsm.context import FSMContext
from aiogram.enums import ChatAction
from aiogram.utils.markdown import hcode
from dotenv import load_dotenv

# OpenAI (async) SDK
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY .env da topilmadi")

# OpenAI klient
oai = AsyncOpenAI(api_key=OPENAI_API_KEY)

echo_router = Router()

# ---- Rate limit boshqaruvi: parallel so'rovlarni yumshatamiz ----
# Bir vaqtning o'zida OpenAI'ga maksimal 4 ta so'rov ketadi (xohlasangiz oshiring/yoki kamaytiring)
OAI_SEMAPHORE = asyncio.Semaphore(4)

# ---- Juda sodda xotira (RAM) — har user uchun oxirgi 8 xabar ----
HISTORY: Dict[int, Deque[ChatCompletionMessageParam]] = defaultdict(lambda: deque(maxlen=8))
SYSTEM_PROMPT = (
    "You are a helpful assistant for a Telegram bot. "
    "If the user asks for code or it makes sense to show code, reply with a fenced code block ```lang ...```. "
    "Keep answers clear, concise, and well-formatted."
)

# ---- Telegram limitlari uchun javobni bo'lib yuborish ----
def tg_chunks(text: str, limit: int = 3900) -> List[str]:
    parts: List[str] = []
    for para in text.split("\n\n"):
        if len(para) <= limit:
            parts.append(para)
        else:
            parts.extend(wrap(para, width=limit))
    return [p for p in parts if p.strip()] or ["(bo'sh javob)"]

# ---- Model javobida code-blocklarni aniqlash ----
CODE_BLOCK_RE = re.compile(r"```([\w+-]*)\n(.*?)\n```", re.DOTALL)

def split_text_and_code(content: str) -> List[Tuple[str, Optional[str]]]:
    """
    Model javobini bo'laklarga ajratadi.
    Har bo'lak (chunk) (matn, lang) ko'rinishida: lang=None bo'lsa oddiy matn, aks holda code.
    """
    blocks: List[Tuple[str, Optional[str]]] = []
    idx = 0
    for m in CODE_BLOCK_RE.finditer(content):
        start, end = m.span()
        lang = (m.group(1) or '').strip() or None
        code = m.group(2)
        # oldingi matn
        if start > idx:
            text_part = content[idx:start].strip()
            if text_part:
                blocks.append((text_part, None))
        # code bloki
        blocks.append((code, lang))
        idx = end
    # qolgan matn
    if idx < len(content):
        rest = content[idx:].strip()
        if rest:
            blocks.append((rest, None))
    return blocks or [(content, None)]

async def chatgpt_answer(user_id: int, prompt: str) -> str:
    """
    ChatGPT'dan javob: 429 Too Many Requests bo'lsa bir necha marta retry qiladi.
    Kontekstni (HISTORY) hisobga oladi.
    """
    # Kichik throttling (yig'ilgan burstlarni yumshatish uchun)
    await asyncio.sleep(0.15)

    # exponential backoff parametrlari
    max_retries = 3
    backoff = 0.7  # sekund

    # Suhbat xabarlari
    messages: List[ChatCompletionMessageParam] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(HISTORY[user_id])
    messages.append({"role": "user", "content": prompt})

    async with OAI_SEMAPHORE:
        for attempt in range(1, max_retries + 1):
            try:
                resp = await oai.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=messages,
                    temperature=0.35,
                    max_tokens=700,
                )
                answer = resp.choices[0].message.content or "Uzr, javob topilmadi."
                # xotiraga qo'shamiz
                HISTORY[user_id].append({"role": "user", "content": prompt})
                HISTORY[user_id].append({"role": "assistant", "content": answer})
                return answer
            except Exception as e:
                status = getattr(e, "status_code", None)
                if status == 429 and attempt < max_retries:
                    await asyncio.sleep(backoff)
                    backoff *= 1.8
                    continue
                raise

async def send_pretty_answer(message: types.Message, content: str) -> None:
    """Javobni bo'laklarga ajratib, matn va kodni alohida chiroyli yuboradi."""
    blocks = split_text_and_code(content)
    for block, lang in blocks:
        if lang:  # code block
            # Agar juda uzun bo'lsa, parchalaymiz.
            # Kodni bo'lib yuborishda ham code fence saqlanadi.
            parts = tg_chunks(block, limit=3600)
            for i, part in enumerate(parts, start=1):
                header = f"```{lang}\n{part}\n```"
                await message.answer(header)
        else:
            for part in tg_chunks(block):
                await message.answer(part)

# ----------------- Handlers -----------------

@echo_router.message(Command("reset"))
async def reset_ctx(message: types.Message):
    HISTORY.pop(message.from_user.id, None)
    await message.reply("♻️ Kontekst tozalandi. Yangi suhbatni boshladik.")

@echo_router.message(F.text, StateFilter(None))
async def bot_echo(message: types.Message):
    """
    Hech qanday state yo'q — foydalanuvchi matnini ChatGPTga yuboramiz va javobini qaytaramiz.
    """
    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    user_text = message.text

    try:
        answer = await chatgpt_answer(message.from_user.id, user_text)
    except Exception:
        answer = "⚠️ API bilan bog'lanishda muammo yuz berdi. Bir ozdan so'ng qayta urinib ko'ring."

    await send_pretty_answer(message, answer)

@echo_router.message(F.text)
async def bot_echo_all(message: types.Message, state: FSMContext):
    """
    Foydalanuvchi state ichida — ChatGPTga yuboramiz (xohlasangiz state nomini promptga qo'shishingiz mumkin).
    """
    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    user_text = message.text

    # ixtiyoriy (agar state nomini kontekstga qo'shmoqchi bo'lsangiz):
    # state_name = await state.get_state() or "unknown"
    # user_text = f"[state={state_name}] {user_text}"

    try:
        answer = await chatgpt_answer(message.from_user.id, user_text)
    except Exception:
        answer = "⚠️ API bilan bog'lanishda muammo yuz berdi. Bir ozdan so'ng qayta urinib ko'ring."

    await send_pretty_answer(message, answer)

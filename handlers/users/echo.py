# guruh_yordamchi_echo.py — IT TAT guruhlari uchun aqlli, qisqa, hazilkash yordamchi
# -----------------------------------------------------------------------------
# Xususiyatlar:
#  - GPT-4o mini (yoki .env dagi OPENAI_MODEL) bilan ishlaydi
#  - Faqat guruh(lar)da javob beradi; private chatda jim
#  - @mention, /ask, ustoz (@davronovsimple) signali va qizigan bahs holatlarida qisqa javob
#  - IT/dasturlash doirasidan tashqariga chiqmaydi (filter)
#  - "Kod yozing" degan so'rovlarni hazil bilan rad etadi; hech qachon kod bermaydi
#  - Ustozga hurmat: "To'g'ri aytyapsiz ustoz" + 1 qisqa qo'shimcha
#  - Guruh va foydalanuvchi bo'yicha rate-limit + guruh-cooldown (45–90 s)
#  - Kontekst xotira (har chat uchun oxirgi 8 xabar)
#  - /reset (kontekstni tozalash), /tip (kunlik bitta maslahat), /xulosa (3 punktli xulosa)
#  - Retry + exponential backoff (429/vaqtinchalik xatolar uchun)
#  - Javoblar doimo juda qisqa: 1–2 jumla (eng ko'pi 3)

import os
import re
import asyncio
import random
import time
from collections import defaultdict, deque
from typing import Deque, Dict, List, Tuple, Optional

from aiogram import types, Router, F
from aiogram.filters import StateFilter, Command
from aiogram.fsm.context import FSMContext
from aiogram.enums import ChatAction, ChatType
from dotenv import load_dotenv

# OpenAI (async) SDK
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
BOT_USERNAME = os.getenv("BOT_USERNAME", "@chaqqonaibot")  # masalan: 'guruh_yordamchi_bot'
USTOZ_USERNAME = os.getenv("USTOZ_USERNAME", "davronovsimple")  # '@' siz

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY .env da topilmadi")

oai = AsyncOpenAI(api_key=OPENAI_API_KEY)

echo_router = Router()

# ---------------------- Konfiguratsiya ----------------------
ALLOWED_TOPICS_RE = re.compile(
    r"\b(python|django|fastapi|aiogram|javascript|typescript|react|tailwind|html|css|sass|less|\bjs\b|ts|node|npm|yarn|vite|webpack|rollup|jest|pytest|unittest|docker|compose|linux|bash|zsh|git|github|gitlab|ci/cd|postgres|mysql|sqlite|redis|celery|rabbitmq|nginx|gunicorn|uvicorn|daphne|supervisor|systemd|selenium|playwright|rest|graphql|drf|jwt|oauth|websocket|http|api|regex|typing|oop|etl|pandas|numpy|matplotlib|plotly|k8s|kubernetes|helm|terraform)\b",
    re.I,
)
HEATED_MARKERS = ("yo'q", "noto'g'ri", "ishlamayapti", "xato", "error", "qanday", "nega", "qayerda")

# Rate limits
USER_RATE_WINDOW = int(os.getenv("USER_RATE_WINDOW", 20))   # s
GROUP_COOLDOWN_MIN = int(os.getenv("GROUP_COOLDOWN_MIN", 45))
GROUP_COOLDOWN_MAX = int(os.getenv("GROUP_COOLDOWN_MAX", 90))

# Parallel OpenAI so'rovlarini cheklash
OAI_SEMAPHORE = asyncio.Semaphore(int(os.getenv("OAI_PARALLEL", 4)))

# Xotira: har chat uchun oxirgi 8 xabar
HISTORY: Dict[int, Deque[ChatCompletionMessageParam]] = defaultdict(lambda: deque(maxlen=8))

# Foydalanuvchi va guruh throttling holati
LAST_SEEN_USER: Dict[int, float] = defaultdict(float)   # user_id -> ts
GROUP_COOLDOWN: Dict[int, float] = defaultdict(float)   # chat_id -> ts
THREAD_COUNT: Dict[Tuple[int, Optional[int]], int] = defaultdict(int)

# ---------------------- Prompt (persona) ----------------------
SYSTEM_PROMPT = (
    "Siz 'GuruhYordamchi' nomli Telegram guruh assistentisiz. Auditoriya — IT TATdagi dasturchi talabalar. "
    "Rolingiz: faqat IT/dasturlash mavzularida juda qisqa va lo'nda javob bering. Doimo hurmatli, yengil hazil bilan. "
    "Hech qachon kod yozmaysiz, hatto so'ralsa ham. Kod o'rniga yo'nalish, tekshirish ro'yxati yoki konseptual maslahat bering. "
    "Agar kimligingiz so'ralsa: 'Men ustoz Davronov G'olibjonning yordamchisiman, IT TAT o'quv markazidan.' deb javob bering. "
    "Ustoz (@davronovsimple) fikr bildirsa, avval tasdiq: 'To'g'ri aytyapsiz ustoz ✅', so'ng bitta juda qisqa qo'shimcha bering. "
    "Mavzu doirasi: python, js, web, backend, frontend, devops va shu kabi. Boshqa mavzularni qatiy rad eting. "
    "Javob formati: 1–2 jumla (eng ko'pi 3) yoki 3 punktli juda qisqa bullet. Emojilar minimal (🙂, ✅) va faqat kerak bo'lsa. "
    "Kod so'ralsa hazil bilan yumshoq rad eting (masalan: 'Men kod yozsam, siz nima o'rganasiz? 🙂 Yo'nalish: ...')."
)

# ---------------------- Util funksiyalar ----------------------

def is_group(chat_type: ChatType) -> bool:
    return chat_type in {ChatType.GROUP, ChatType.SUPERGROUP}

async def get_admin_ids(bot, chat_id: int) -> set:
    admins = await bot.get_chat_administrators(chat_id)
    return {adm.user.id for adm in admins}

def is_from_ustoz(admin_ids: set, user_id: int, text: str) -> bool:
    if user_id in admin_ids:
        return True
    return f"@{USTOZ_USERNAME}" in (text or "")

def looks_it_topic(text: str) -> bool:
    return bool(ALLOWED_TOPICS_RE.search(text or ""))

def is_question_or_confusion(text: str) -> bool:
    low = (text or "").lower()
    return ("?" in low) or any(k in low for k in ("qanday", "nega", "nimaga", "ishlamayapti", "xato", "error"))

def is_heating_up(chat_id: int, thread_id: Optional[int], text: str) -> bool:
    key = (chat_id, thread_id)
    THREAD_COUNT[key] += 1
    low = (text or "").lower()
    hot = sum(1 for k in HEATED_MARKERS if k in low)
    return THREAD_COUNT[key] >= 3 or hot >= 2

def user_rate_limited(user_id: int) -> bool:
    now = time.time()
    if now - LAST_SEEN_USER[user_id] < USER_RATE_WINDOW:
        return True
    LAST_SEEN_USER[user_id] = now
    return False

def group_cooldown_ok(chat_id: int) -> bool:
    now = time.time()
    window = random.randint(GROUP_COOLDOWN_MIN, GROUP_COOLDOWN_MAX)
    if now - GROUP_COOLDOWN[chat_id] >= window:
        GROUP_COOLDOWN[chat_id] = now
        return True
    return False

MENTION_WORDS = ("fikir", "izoh", "aniqlashtir", "xulosa")

def should_respond(message: types.Message, admin_ids: set) -> bool:
    text = message.text or ""
    # 1) @mention yoki /ask
    if BOT_USERNAME and f"@{BOT_USERNAME}" in text:
        return True
    if text.startswith("/ask"):
        return True
    # 2) Ustoz/admin signali + savol/aniqlik
    if is_from_ustoz(admin_ids, message.from_user.id, text) and is_question_or_confusion(text):
        return True
    # 3) Qizigan bahs + IT mavzusi + guruh cooldown
    if looks_it_topic(text) and is_question_or_confusion(text) and is_heating_up(message.chat.id, message.message_thread_id, text):
        return group_cooldown_ok(message.chat.id)
    # 4) Trigger so'zlar
    if looks_it_topic(text) and any(w in text.lower() for w in MENTION_WORDS):
        return group_cooldown_ok(message.chat.id)
    return False

# ---------------------- OpenAI mantiqi ----------------------
async def chatgpt_answer(chat_id: int, prompt: str) -> str:
    # juda qisqa delay burstlarni yumshatish uchun
    await asyncio.sleep(0.12)

    max_retries = 3
    backoff = 0.7

    messages: List[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    # Chatga bog'lab kontekst yuritamiz
    messages.extend(HISTORY[chat_id])
    messages.append({"role": "user", "content": prompt})

    async with OAI_SEMAPHORE:
        for attempt in range(1, max_retries + 1):
            try:
                resp = await oai.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=450,
                )
                answer = resp.choices[0].message.content or "Uzr, javob topilmadi."
                # Xotiraga yozamiz (chat bo'yicha)
                HISTORY[chat_id].append({"role": "user", "content": prompt})
                HISTORY[chat_id].append({"role": "assistant", "content": answer})
                return sanitize_answer(answer)
            except Exception as e:
                status = getattr(e, "status_code", None)
                if status == 429 and attempt < max_retries:
                    await asyncio.sleep(backoff)
                    backoff *= 1.8
                    continue
                raise

CODE_FENCE_RE = re.compile(r"```[\s\S]*?```", re.M)

def sanitize_answer(ans: str) -> str:
    """Hech qachon kod yubormaslik: code fence bo'lsa, hazil bilan almashtiramiz va qisqartiramiz."""
    if CODE_FENCE_RE.search(ans):
        ans = CODE_FENCE_RE.sub("Men kod yozsam, siz nima o'rganasiz? 🙂 Yo'nalish: 1) muammoni aniqlang; 2) kichik misol; 3) log/xatoni o'qing.", ans)
    # Juda uzun gaplarni qisqartirish: 3 jumladan yig'ish
    sentences = re.split(r"(?<=[.!?])\s+", ans.strip())
    if len(sentences) > 3:
        ans = " ".join(sentences[:3])
    return ans.strip() or "(bo'sh javob)"

# ---------------------- Handlers ----------------------

@echo_router.message(Command("reset"))
async def reset_ctx(message: types.Message):
    HISTORY.pop(message.chat.id, None)
    await message.reply("♻️ Kontekst tozalandi. Yangi suhbat boshlandi.")

@echo_router.message(Command("tip"))
async def tip_cmd(message: types.Message):
    tips = [
        "Minimal reproduksiya — tez yechimning kaliti. ✅",
        "Commitlarni kichik qiling, revert oson bo'ladi.",
        "Loglarni o'qing: xato odatda u yerda qichqiradi 🙂",
        "Env farqlarini tekshiring: lokal ≠ server.",
        "Test yozish — ertangi tinchlik 🙂",
    ]
    await message.reply(random.choice(tips))

@echo_router.message(Command("xulosa"))
async def xulosa_cmd(message: types.Message):
    # Chatdagi oxirgi kontekstdan 3 punktli xulosa so'raymiz
    prompt = (
        "Quyidagi kontekstdan 3 punktlik juda qisqa xulosa yozing. Uzun matn yozmang.\n"
        "Formati: 1) muammo 2) asosiy fikr 3) keyingi qadam."
    )
    try:
        ans = await chatgpt_answer(message.chat.id, prompt)
    except Exception:
        ans = "⚠️ AI bilan ulanishda muammo."
    await message.reply(ans)

@echo_router.message(F.text, StateFilter(None))
async def group_only_listener(message: types.Message):
    # 0) Faqat guruh
    if not is_group(message.chat.type):
        return  # private chatda javob bermaymiz

    text = message.text or ""

    # 1) IT mavzusi filtri (katta oqimni kamaytirish)
    if not looks_it_topic(text) and not (BOT_USERNAME and f"@{BOT_USERNAME}" in text) and not text.startswith("/ask"):
        return

    # 2) Adminlarni oldindan topib olamiz
    admin_ids = await get_admin_ids(message.bot, message.chat.id)

    # 3) Javob berish shartlari
    if not should_respond(message, admin_ids):
        return

    # 4) Per-user rate limit
    if user_rate_limited(message.from_user.id):
        return

    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)

    # /ask komandasi bo'lsa, matndan olib tashlaymiz
    prompt = text
    if prompt.startswith("/ask"):
        prompt = prompt[len("/ask"):].strip()

    # Ustozga hurmatli prefiks qo'shish (agar xabar ustozdan bo'lsa)
    if is_from_ustoz(admin_ids, message.from_user.id, text) and f"@{USTOZ_USERNAME}" in text:
        prompt = "Ustoz fikrini tasdiqlab, bitta qisqa qo'shimcha bering. " + prompt

    try:
        answer = await chatgpt_answer(message.chat.id, prompt)
    except Exception:
        answer = "⚠️ API bilan bog'lanishda muammo yuz berdi. Birozdan so'ng urinib ko'ring."

    # Qisqartirish (extra guard)
    if len(answer) > 350:
        answer = answer[:330] + "..."

    await message.reply(answer, disable_web_page_preview=True)

# Foydalanuvchi state holatida bo'lsa ham xuddi shu mantiq
@echo_router.message(F.text)
async def group_only_listener_with_state(message: types.Message, state: FSMContext):
    await group_only_listener(message)

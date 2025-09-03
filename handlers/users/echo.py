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
BOT_USERNAME = os.getenv("BOT_USERNAME", "chaqqonaibot")  # masalan: 'guruh_yordamchi_bot'
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
# Paste this over your broken SYSTEM_PROMPT block
# Clean, single string; no stray quotes; tight instructions.

SYSTEM_PROMPT = (
    "Siz 'GuruhYordamchi' nomli Telegram guruh assistentisiz. Auditoriya — IT TATdagi dasturchi talabalar. "
    "Faqat trigger bo'lganda javob bering (mention, /ask, ustoz xabari, qizigan bahs yoki bot javobiga reply). "
    "Faqat IT/dasturlash doirasida juda qisqa va lo'nda yozing: 1–2 jumla (eng ko'pi 3) yoki 3 punkt. "
    "Hech qachon kod yozmaysiz. Kod so'ralsa, qisqa hazil bilan rad eting va yo'nalish bering. "
    "Hech qachon o'zingizni tanishtirmang — faqat aniq so'ralganda: 'Men ustoz Davronov G'olibjonning yordamchisiman, IT TAT o'quv markazidan.' deb javob bering. "
    "Ustoz (@davronovsimple) xabar yozsa, avval tasdiqlang: 'To'g'ri aytyapsiz ustoz ✅', so'ng bitta juda qisqa qo'shimcha bering. "
    "Boshqa mavzularni qat'iy rad eting. Emojilar minimal (🙂, ✅) va faqat kerak bo'lsa."
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
    # tezroq tutish: 2 xabar yoki 1 ta "issiq" so'z yetarli
    return THREAD_COUNT[key] >= 2 or hot >= 1

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

MENTION_WORDS = ("fikir", "izoh", "aniqlashtir", "xulosa", "fikr")

# --- qo'shimcha trigger yordamchilari ---
def mentioned_directly(text: str) -> bool:
    return bool(BOT_USERNAME) and (f"@{BOT_USERNAME}".lower() in (text or "").lower())

def reply_to_bot(message: types.Message) -> bool:
    u = getattr(getattr(message, "reply_to_message", None), "from_user", None)
    if not u or not u.username:
        return False
    return bool(BOT_USERNAME) and (u.username.lower() == BOT_USERNAME.lower())

def is_ustoz_message(message: types.Message) -> bool:
    u = message.from_user
    return bool(u and u.username and (u.username.lower() == USTOZ_USERNAME.lower()))

# "kod yozib ber" tipidagi so'rovlarni oldindan ushlaymiz (oddiy heuristika)
CODE_HINTS = ("kod yoz", "kodini yoz", "kod yozib ber", "write code", "code sample", "snippet", "namuna")

def is_code_request(text: str) -> bool:
    low = (text or "").lower()
    return any(h in low for h in CODE_HINTS)

def should_respond(message: types.Message, admin_ids: set) -> bool:
    text = message.text or ""
    # 0) agar botning javobiga reply bo'lsa
    if reply_to_bot(message):
        return True
    # 1) @mention yoki /ask
    if mentioned_directly(text) or text.startswith("/ask"):
        return True
    # 2) ustoz xabari (IT bo'lsa)
    if is_ustoz_message(message) and looks_it_topic(text):
        return True
    # 3) qizigan bahs + IT + savol/aniqlik + cooldown
    if looks_it_topic(text) and is_question_or_confusion(text) and is_heating_up(message.chat.id, message.message_thread_id, text):
        return group_cooldown_ok(message.chat.id)
    # 4) trigger so'zlar
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

def sanitize_answer(ans: str, prompt: str) -> str:
    """Kod bloklarini chiqarib yubormaslik; o'zini tanishtirishni faqat so'ralganda; javobni juda qisqa tutish."""
    # Kod bloklarini hazil bilan almashtiramiz
    if CODE_FENCE_RE.search(ans):
        ans = CODE_FENCE_RE.sub("Men kod yozsam, siz nima o'rganasiz? 🙂 Yo'nalish: 1) muammoni aniqlang; 2) kichik misol; 3) log/xatoni o'qing.", ans)
    # O'zini tanishtirishni faqat "kim?" so'ralganda qoldiramiz
    WHOAMI_SENT = "Men ustoz Davronov G'olibjonning yordamchisiman"
    WHOASK_HINTS = ("kimsan", "kimligi", "kim o'zi", "who are you")
    lowp = (prompt or "").lower()
    if WHOAMI_SENT in ans and not any(h in lowp for h in WHOASK_HINTS):
        ans = ans.replace(WHOAMI_SENT, "").strip()
    # Juda uzun bo'lsa — 3 jumlagacha qisqartiramiz (sodda bo'luvchi)
    buf = ans.strip()
    for sep in [". ", "? ", "! "]:
        buf = buf.replace(sep, "|SEP|")
    sentences = [s.strip() for s in buf.split("|SEP|") if s.strip()]
    if len(sentences) > 3:
        ans = ". ".join(sentences[:3])
    else:
        ans = ". ".join(sentences)
    return ans.strip() or "(bo'sh javob)"

# ---------------------- Util: playful mode ----------------------
PLAYFUL_HINTS = (
    "hazil", "kuldir", "mem", "katta ketma", "uka", "sen mening yordamchimisan", "sening vazifang", "botjon", "bot aka", "aka bot", "bro", "aka"
)

def is_playful_request(text: str) -> bool:
    low = (text or "").lower()
    return any(h in low for h in PLAYFUL_HINTS)

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
    if not is_group(message.chat.type):
        return  # faqat guruh

    text = message.text or ""

    # Mention/savol/IT yoki reply-to-bot bo'lmasa — jim (lekin hazilga bitta qisqa javob berishimiz mumkin)
    if not (looks_it_topic(text) or is_question_or_confusion(text) or mentioned_directly(text)
            or text.startswith("/ask") or reply_to_bot(message)):
        if is_playful_request(text) and group_cooldown_ok(message.chat.id):
            jokes = [
                "Katta ketmang, uka 🙂 vazifa sodda: savolni aniqlang, keyin bir qadam qilib sinang.",
                "Bot aka hushyor! Ammo IT bo'lmasa, jim turaman 😉",
                "Hazil yaxshi, lekin commitlar yanayam yaxshi 😄",
            ]
            await message.reply(random.choice(jokes))
        return

    admin_ids = await get_admin_ids(message.bot, message.chat.id)
    if not should_respond(message, admin_ids):
        return

    if user_rate_limited(message.from_user.id):
        return

    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)

    # /ask bo'lsa prefiksini olib tashlaymiz
    prompt = text[len("/ask"):].strip() if text.startswith("/ask") else text

    # Agar off-topic bo'lsa, ammo bevosita murojaat (mention/reply) yoki hazil bo'lsa — API'ga chiqmasdan qisqa hazil
    if (not looks_it_topic(prompt) and (mentioned_directly(text) or reply_to_bot(message))) or is_playful_request(prompt):
        jokes = [
            "Katta ketmang, uka 🙂 Men IT mavzularida kuchliman. Savol bo'lsa marhamat",
            "Ha, yordamchingizman — lekin darsdan qochirmayman 😉 Savolingizni aniqlashtiring",
            "Hazil joyida, lekin kodni o'zingiz yozasiz 😄 Yo'nalish beraman!",
        ]
        await message.reply(random.choice(jokes))
        return

    # Kod so'rovi — APIsiz, darrov hazil
    if is_code_request(prompt):
        jokes = [
            "Men kod yozsam, siz nima o'rganasiz? 🙂 Yo'nalish: muammoni aniqlang, kichik misol qiling, logni o'qing.",
            "Kod? Yo‘q-yo‘q 🙂 Avval fikrni aniqla, o‘zing urin — men yo‘nalishni aytaman.",
            "Kodni ustozlar yozadi, o‘quvchi esa o‘rganadi 😉 Qadam: talab → skelet → sinov.",
        ]
        await message.reply(random.choice(jokes))
        return

    # Ustoz xabari bo'lsa — doim tasdiq + bitta lo'nda qo'shimcha
    if is_ustoz_message(message):
        prompt = "Ustoz fikrini qisqa tasdiqlab, bitta lo'nda qo'shimcha bering: " + prompt

    try:
        answer = await chatgpt_answer(message.chat.id, prompt)
    except Exception:
        answer = "⚠️ API bilan bog'lanishda muammo yuz berdi. Birozdan so'ng urinib ko'ring."

    if len(answer) > 350:
        answer = answer[:330] + "..."

    await message.reply(answer, disable_web_page_preview=True)

@echo_router.message(F.text)
async def group_only_listener_with_state(message: types.Message, state: FSMContext):
    await group_only_listener(message)

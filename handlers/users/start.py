from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.types import Message
from html import escape

user_router = Router()

@user_router.message(CommandStart())
async def user_start(message: Message):
    # Foydalanuvchi nomini xavfsiz ko‘rsatish (HTML escapе)
    name = escape((message.from_user.full_name or message.from_user.first_name or "do‘st").strip())

    welcome_message = (
        f"🤖 <b>Chaqqon AI</b> ga xush kelibsiz, {name}!<br>"
        "⚡️ Savolingizni yozing — chaqqon va aniq javob beraman.<br><br>"
        "🧹 Kontekstni tozalash: <code>/reset</code><br>"
        "💬 Har qanday fikr-mulohazani bemalol yozing."
    )

    await message.reply(welcome_message, parse_mode="HTML")

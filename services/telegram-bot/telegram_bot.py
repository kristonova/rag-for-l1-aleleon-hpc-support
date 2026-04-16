#!/usr/bin/env python3
"""
telegram_bot.py — Telegram Bot untuk RAG ALELEON HPC
Menerima pertanyaan dari user Telegram, kirim ke RAG API, lalu balas jawabannya.
"""

import os
import asyncio
import logging
import requests
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# ── Konfigurasi ──────────────────────────────────────────────
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
RAG_API_URL = os.getenv("RAG_API_URL", "http://rag-api:8080")

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Animasi placeholder saat menunggu jawaban RAG ────────────
PROGRESS_FRAMES = [
    "⏳ Sedang mencari jawaban...",
    "🔍 Mencari di dokumentasi wiki...",
    "🤖 Memproses dengan AI...",
    "💡 Menyusun jawaban...",
]


# ── Handler: /start ──────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Pesan selamat datang saat user pertama kali chat."""
    welcome = (
        "👋 Halo! Saya bot asisten HPC ALELEON.\n\n"
        "Kirim pertanyaan tentang ALELEON/Slurm/HPC, "
        "dan saya akan menjawab berdasarkan dokumentasi wiki.\n\n"
        "Contoh:\n"
        "• Bagaimana cara membuat conda environment?\n"
        "• Berapa kuota storage HOME untuk akun perseorangan?\n"
        "• Cara submit batch job di Slurm?\n\n"
        "Ketik pertanyaan kamu langsung! \U0001f680"
    )
    await update.message.reply_text(welcome)


# ── Handler: /help ───────────────────────────────────────────
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Tampilkan bantuan."""
    help_text = (
        "📖 **Cara pakai bot ini:**\n\n"
        "Cukup ketik pertanyaan kamu langsung di chat.\n"
        "Bot akan mencari jawaban dari wiki ALELEON.\n\n"
        "**Perintah:**\n"
        "/start — Pesan selamat datang\n"
        "/help — Bantuan ini\n"
        "/status — Cek status RAG API"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")


# ── Handler: /status ─────────────────────────────────────────
async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Cek apakah RAG API berjalan."""
    try:
        resp = requests.get(f"{RAG_API_URL}/health", timeout=10)
        data = resp.json()
        emoji = "✅" if data.get("status") == "ready" else "⏳"
        await update.message.reply_text(f"{emoji} RAG API status: {data.get('status', 'unknown')}")
    except Exception as e:
        await update.message.reply_text(f"❌ RAG API tidak bisa dihubungi: {e}")


# ── Background: Typing indicator + animasi placeholder ──────
async def _keep_alive_indicator(chat, placeholder_msg, stop_event: asyncio.Event):
    """
    Background task yang berjalan selama menunggu respons RAG:
    1. Kirim ulang 'typing' action setiap 4 detik (Telegram menghilangkan
       indikator typing setelah ~5 detik).
    2. Putar animasi teks placeholder agar user tahu proses berjalan.
    """
    frame_idx = 0
    while not stop_event.is_set():
        # Kirim typing indicator
        try:
            await chat.send_action("typing")
        except Exception:
            pass

        # Update teks placeholder (cycle melalui PROGRESS_FRAMES)
        frame_idx = (frame_idx + 1) % len(PROGRESS_FRAMES)
        try:
            await placeholder_msg.edit_text(PROGRESS_FRAMES[frame_idx])
        except Exception:
            pass

        # Tunggu 4 detik atau sampai stop_event di-set
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=4.0)
            break  # stop_event sudah di-set, keluar
        except asyncio.TimeoutError:
            continue  # Belum selesai, lanjut loop


# ── Handler: Pesan teks biasa (pertanyaan) ───────────────────
async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Kirim pertanyaan user ke RAG API dan balas jawabannya."""
    question = update.message.text
    user = update.effective_user
    logger.info(f"Pertanyaan dari {user.first_name} (@{user.username}): {question}")

    chat = update.message.chat

    # 1) Kirim typing + placeholder message
    await chat.send_action("typing")
    placeholder_msg = await update.message.reply_text(PROGRESS_FRAMES[0])

    # 2) Mulai background task: typing loop + animasi placeholder
    stop_event = asyncio.Event()
    indicator_task = asyncio.create_task(
        _keep_alive_indicator(chat, placeholder_msg, stop_event)
    )

    try:
        # 3) Kirim ke RAG API (blocking I/O di thread terpisah agar tidak freeze)
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: requests.post(
                f"{RAG_API_URL}/ask",
                json={"question": question},
                timeout=300,  # LLM bisa lambat, beri waktu 5 menit
            ),
        )
        resp.raise_for_status()
        data = resp.json()

        # 4) Stop animasi
        stop_event.set()
        await indicator_task

        answer = data.get("answer", "Maaf, saya tidak bisa menjawab saat ini.")
        sources = data.get("sources", [])

        # Format jawaban
        reply = f"{answer}\n"

        if sources:
            reply += "\n📚 *Sumber:*\n"
            for i, src in enumerate(sources, 1):
                label = src.get("title", "Unknown")
                section = src.get("section", "")
                url = src.get("source_url", "")
                if section:
                    label += f" → {section}"
                reply += f"{i}. [{label}]({url})\n"
                # Tambahkan justifikasi jika ada
                justification = src.get("justification", "")
                if justification:
                    reply += f"   _💡 {justification}_\n"

        # 5) Edit placeholder dengan jawaban final
        try:
            if len(reply) > 4000:
                # Jawaban lebih dari 4000 karakter → edit placeholder + kirim sumber terpisah
                await placeholder_msg.edit_text(answer[:4000], parse_mode="Markdown")
                if sources:
                    source_text = "📚 *Sumber:*\n"
                    for i, src in enumerate(sources, 1):
                        label = src.get("title", "Unknown")
                        url = src.get("source_url", "")
                        source_text += f"{i}. [{label}]({url})\n"
                    await update.message.reply_text(source_text, parse_mode="Markdown")
            else:
                await placeholder_msg.edit_text(reply, parse_mode="Markdown")
        except Exception as parse_err:
            logger.warning(f"Gagal parse Markdown, fallback ke plain text: {parse_err}")
            if len(reply) > 4000:
                await placeholder_msg.edit_text(answer[:4000])
                if sources:
                    source_text = "📚 Sumber:\n"
                    for i, src in enumerate(sources, 1):
                        label = src.get("title", "Unknown")
                        url = src.get("source_url", "")
                        source_text += f"{i}. {label} - {url}\n"
                    await update.message.reply_text(source_text)
            else:
                await placeholder_msg.edit_text(reply)

    except requests.Timeout:
        stop_event.set()
        await indicator_task
        await placeholder_msg.edit_text(
            "⏰ Maaf, RAG membutuhkan waktu terlalu lama. Coba lagi nanti."
        )
    except requests.ConnectionError:
        stop_event.set()
        await indicator_task
        await placeholder_msg.edit_text(
            "❌ Tidak bisa terhubung ke RAG API. Pastikan service berjalan."
        )
    except Exception as e:
        stop_event.set()
        await indicator_task
        logger.error(f"Error: {e}")
        await placeholder_msg.edit_text(
            f"⚠️ Terjadi error: {str(e)[:200]}"
        )


# ── Main ─────────────────────────────────────────────────────
def main():
    if not TELEGRAM_TOKEN:
        print("❌ ERROR: Set environment variable TELEGRAM_TOKEN terlebih dahulu!")
        print("   Contoh: export TELEGRAM_TOKEN='7123456789:AAH...'")
        return

    print(f"🤖 Memulai Telegram Bot...")
    print(f"   RAG API URL: {RAG_API_URL}")

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Daftarkan handler
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))

    # Jalankan bot (long polling)
    print("✅ Bot berjalan! Tekan Ctrl+C untuk berhenti.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()


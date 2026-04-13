#!/usr/bin/env python3
"""
telegram_bot.py — Telegram Bot untuk RAG ALELEON HPC
Menerima pertanyaan dari user Telegram, kirim ke RAG API, lalu balas jawabannya.
"""

import os
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


# ── Handler: Pesan teks biasa (pertanyaan) ───────────────────
async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Kirim pertanyaan user ke RAG API dan balas jawabannya."""
    question = update.message.text
    user = update.effective_user
    logger.info(f"Pertanyaan dari {user.first_name} (@{user.username}): {question}")

    # Kirim indikator "sedang mengetik"
    await update.message.chat.send_action("typing")

    try:
        # Kirim ke RAG API
        resp = requests.post(
            f"{RAG_API_URL}/ask",
            json={"question": question},
            timeout=120,  # LLM bisa lambat, beri waktu 2 menit
        )
        resp.raise_for_status()
        data = resp.json()

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

        # Telegram punya limit 4096 karakter per pesan
        try:
            if len(reply) > 4000:
                # Potong jawaban dan kirim sumber terpisah
                await update.message.reply_text(answer[:4000], parse_mode="Markdown")
                if sources:
                    source_text = "📚 *Sumber:*\n"
                    for i, src in enumerate(sources, 1):
                        label = src.get("title", "Unknown")
                        url = src.get("source_url", "")
                        source_text += f"{i}. [{label}]({url})\n"
                    await update.message.reply_text(source_text, parse_mode="Markdown")
            else:
                await update.message.reply_text(reply, parse_mode="Markdown")
        except Exception as parse_err:
            logger.warning(f"Gagal parse Markdown, fallback ke plain text: {parse_err}")
            if len(reply) > 4000:
                await update.message.reply_text(answer[:4000])
                if sources:
                    source_text = "📚 Sumber:\n"
                    for i, src in enumerate(sources, 1):
                        label = src.get("title", "Unknown")
                        url = src.get("source_url", "")
                        source_text += f"{i}. {label} - {url}\n"
                    await update.message.reply_text(source_text)
            else:
                await update.message.reply_text(reply)

    except requests.Timeout:
        await update.message.reply_text(
            "⏰ Maaf, RAG membutuhkan waktu terlalu lama. Coba lagi nanti."
        )
    except requests.ConnectionError:
        await update.message.reply_text(
            "❌ Tidak bisa terhubung ke RAG API. Pastikan service berjalan."
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        await update.message.reply_text(
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

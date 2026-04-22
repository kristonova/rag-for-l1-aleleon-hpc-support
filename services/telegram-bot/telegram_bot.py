#!/usr/bin/env python3
"""
telegram_bot.py — Telegram Bot untuk RAG ALELEON HPC
Menerima pertanyaan dari user Telegram, kirim ke RAG API, lalu balas jawabannya.
Mendukung review skrip Bash/Slurm via paste teks atau file upload.
"""

import os
import asyncio
import logging
import html
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

# Ekstensi file yang di-support untuk review skrip
SCRIPT_EXTENSIONS = {".sh", ".slurm", ".sbatch", ".bash"}

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Animasi placeholder saat menunggu jawaban RAG ────────────
PROGRESS_FRAMES_ASK = [
    "⏳ Sedang mencari jawaban...",
    "🔍 Mencari di dokumentasi wiki...",
    "🤖 Memproses dengan AI...",
    "💡 Menyusun jawaban...",
]

PROGRESS_FRAMES_REVIEW = [
    "⏳ Menerima skrip...",
    "🔍 Menganalisis syntax...",
    "🤖 Memeriksa parameter SBATCH...",
    "💡 Menyusun review...",
]


def escape_html(text: str) -> str:
    """Escape special HTML characters untuk Telegram HTML parse mode."""
    return html.escape(text, quote=False)


def is_shell_script(text: str) -> bool:
    """Deteksi apakah teks yang di-paste adalah skrip Bash/Slurm."""
    indicators = [
        "#!/bin/bash",
        "#!/bin/sh",
        "#!/usr/bin/env bash",
        "#!/usr/bin/env sh",
        "#SBATCH",
    ]
    # Cek minimal 1 indikator di awal atau dalam teks
    text_upper = text.strip()
    for indicator in indicators:
        if indicator in text_upper:
            return True
    return False


def format_reply_html(answer: str, sources: list) -> str:
    """
    Format jawaban + sumber + justifikasi sebagai HTML.
    Telegram HTML parse mode jauh lebih stabil daripada Markdown v1
    karena hanya perlu escape <, >, & di konten teks.
    """
    reply = escape_html(answer) + "\n"

    if sources:
        reply += "\n📚 <b>Sumber:</b>\n"
        for i, src in enumerate(sources, 1):
            title = escape_html(src.get("title", "Unknown"))
            section = src.get("section", "")
            url = src.get("source_url", "")

            label = title
            if section:
                label += f" → {escape_html(section)}"

            if url:
                reply += f'{i}. <a href="{escape_html(url)}">{label}</a>\n'
            else:
                reply += f"{i}. {label}\n"

            # Tambahkan justifikasi jika ada
            justification = src.get("justification", "")
            if justification:
                reply += f"   <i>💡 {escape_html(justification)}</i>\n"

    return reply


def format_sources_html(sources: list) -> str:
    """Format hanya bagian sumber + justifikasi sebagai HTML (untuk pesan terpisah)."""
    source_text = "📚 <b>Sumber:</b>\n"
    for i, src in enumerate(sources, 1):
        title = escape_html(src.get("title", "Unknown"))
        section = src.get("section", "")
        url = src.get("source_url", "")

        label = title
        if section:
            label += f" → {escape_html(section)}"

        if url:
            source_text += f'{i}. <a href="{escape_html(url)}">{label}</a>\n'
        else:
            source_text += f"{i}. {label}\n"

        # Tambahkan justifikasi jika ada
        justification = src.get("justification", "")
        if justification:
            source_text += f"   <i>💡 {escape_html(justification)}</i>\n"

    return source_text


def format_review_html(review: str, issues_found: int, filename: str = None) -> str:
    """Format review skrip sebagai HTML untuk Telegram."""
    header = "🔍 <b>Review Skrip"
    if filename:
        header += f": {escape_html(filename)}"
    header += "</b>\n\n"

    return header + escape_html(review)


# ── Handler: /start ──────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Pesan selamat datang saat user pertama kali chat."""
    welcome = (
        "👋 Halo! Saya bot asisten HPC ALELEON.\n\n"
        "Saya bisa membantu dengan:\n"
        "📖 <b>Pertanyaan</b> — Kirim pertanyaan tentang ALELEON/Slurm/HPC\n"
        "📝 <b>Review Skrip</b> — Kirim/paste skrip .sh untuk saya review\n\n"
        "<b>Contoh pertanyaan:</b>\n"
        "• Bagaimana cara membuat conda environment?\n"
        "• Berapa kuota storage HOME untuk akun perseorangan?\n\n"
        "<b>Review skrip:</b>\n"
        "• Paste langsung skrip yang mengandung #!/bin/bash atau #SBATCH\n"
        "• Upload file .sh / .slurm / .sbatch\n\n"
        "Ketik pertanyaan atau kirim skrip! 🚀"
    )
    await update.message.reply_text(welcome, parse_mode="HTML")


# ── Handler: /help ───────────────────────────────────────────
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Tampilkan bantuan."""
    help_text = (
        "📖 <b>Cara pakai bot ini:</b>\n\n"
        "<b>1. Pertanyaan:</b>\n"
        "Ketik pertanyaan langsung di chat. Bot akan mencari jawaban dari wiki ALELEON.\n\n"
        "<b>2. Review Skrip:</b>\n"
        "• <b>Paste</b> skrip Bash/Slurm langsung di chat (harus mengandung #!/bin/bash atau #SBATCH)\n"
        "• <b>Upload file</b> .sh, .slurm, .sbatch, atau .bash\n"
        "Bot akan mengecek syntax, parameter SBATCH, dan best practice.\n\n"
        "<b>Perintah:</b>\n"
        "/start — Pesan selamat datang\n"
        "/help — Bantuan ini\n"
        "/status — Cek status RAG API"
    )
    await update.message.reply_text(help_text, parse_mode="HTML")


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
async def _keep_alive_indicator(chat, placeholder_msg, stop_event: asyncio.Event, frames=None):
    """
    Background task yang berjalan selama menunggu respons RAG:
    1. Kirim ulang 'typing' action setiap 4 detik.
    2. Putar animasi teks placeholder agar user tahu proses berjalan.
    """
    if frames is None:
        frames = PROGRESS_FRAMES_ASK
    frame_idx = 0
    while not stop_event.is_set():
        try:
            await chat.send_action("typing")
        except Exception:
            pass

        frame_idx = (frame_idx + 1) % len(frames)
        try:
            await placeholder_msg.edit_text(frames[frame_idx])
        except Exception:
            pass

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=4.0)
            break
        except asyncio.TimeoutError:
            continue


# ── Shared: Send reply with HTML formatting ──────────────────
async def _send_html_reply(placeholder_msg, update, reply_html: str):
    """Send formatted HTML reply, with fallback to plain text if HTML parse fails."""
    try:
        if len(reply_html) > 4000:
            # Split: first 4000 chars in placeholder, rest in new message
            await placeholder_msg.edit_text(
                reply_html[:4000], parse_mode="HTML",
                disable_web_page_preview=True,
            )
            remaining = reply_html[4000:]
            if remaining.strip():
                await update.message.reply_text(
                    remaining, parse_mode="HTML",
                    disable_web_page_preview=True,
                )
        else:
            await placeholder_msg.edit_text(
                reply_html, parse_mode="HTML",
                disable_web_page_preview=True,
            )
    except Exception as parse_err:
        logger.warning(f"Gagal parse HTML, fallback ke plain text: {parse_err}")
        # Strip HTML tags for plain text fallback
        import re
        plain = re.sub(r"<[^>]+>", "", reply_html)
        if len(plain) > 4000:
            await placeholder_msg.edit_text(plain[:4000])
        else:
            await placeholder_msg.edit_text(plain)


# ── Handler: Pesan teks biasa (pertanyaan atau skrip) ────────
async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Kirim pertanyaan ke RAG API atau review skrip jika terdeteksi."""
    text = update.message.text
    user = update.effective_user
    chat = update.message.chat

    # Deteksi: skrip atau pertanyaan biasa?
    if is_shell_script(text):
        logger.info(f"Skrip terdeteksi dari {user.first_name} (@{user.username}), routing ke /review-script")
        await _handle_script_review(update, chat, text)
    else:
        logger.info(f"Pertanyaan dari {user.first_name} (@{user.username}): {text[:80]}")
        await _handle_rag_question(update, chat, text)


async def _handle_rag_question(update, chat, question):
    """Jalur RAG: retrieve + generate jawaban dari dokumen."""
    await chat.send_action("typing")
    placeholder_msg = await update.message.reply_text(PROGRESS_FRAMES_ASK[0])

    stop_event = asyncio.Event()
    indicator_task = asyncio.create_task(
        _keep_alive_indicator(chat, placeholder_msg, stop_event, frames=PROGRESS_FRAMES_ASK)
    )

    try:
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: requests.post(
                f"{RAG_API_URL}/ask",
                json={"question": question},
                timeout=300,
            ),
        )
        resp.raise_for_status()
        data = resp.json()

        stop_event.set()
        await indicator_task

        answer = data.get("answer", "Maaf, saya tidak bisa menjawab saat ini.")
        sources = data.get("sources", [])

        for src in sources:
            j = src.get("justification")
            logger.info(f"  Source: {src.get('title')} | justification: {repr(j)}")

        reply = format_reply_html(answer, sources)
        await _send_html_reply(placeholder_msg, update, reply)

    except requests.Timeout:
        stop_event.set()
        await indicator_task
        await placeholder_msg.edit_text("⏰ Maaf, RAG membutuhkan waktu terlalu lama. Coba lagi nanti.")
    except requests.ConnectionError:
        stop_event.set()
        await indicator_task
        await placeholder_msg.edit_text("❌ Tidak bisa terhubung ke RAG API. Pastikan service berjalan.")
    except Exception as e:
        stop_event.set()
        await indicator_task
        logger.error(f"Error: {e}")
        await placeholder_msg.edit_text(f"⚠️ Terjadi error: {str(e)[:200]}")


async def _handle_script_review(update, chat, script_content, filename=None):
    """Jalur review skrip: kirim langsung ke LLM tanpa retrieval."""
    await chat.send_action("typing")
    placeholder_msg = await update.message.reply_text(PROGRESS_FRAMES_REVIEW[0])

    stop_event = asyncio.Event()
    indicator_task = asyncio.create_task(
        _keep_alive_indicator(chat, placeholder_msg, stop_event, frames=PROGRESS_FRAMES_REVIEW)
    )

    try:
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: requests.post(
                f"{RAG_API_URL}/review-script",
                json={"script": script_content},
                timeout=300,
            ),
        )
        resp.raise_for_status()
        data = resp.json()

        stop_event.set()
        await indicator_task

        review = data.get("review", "Maaf, gagal mereview skrip.")
        issues_found = data.get("issues_found", 0)

        reply = format_review_html(review, issues_found, filename=filename)
        await _send_html_reply(placeholder_msg, update, reply)

    except requests.Timeout:
        stop_event.set()
        await indicator_task
        await placeholder_msg.edit_text("⏰ Maaf, review skrip membutuhkan waktu terlalu lama. Coba lagi nanti.")
    except requests.ConnectionError:
        stop_event.set()
        await indicator_task
        await placeholder_msg.edit_text("❌ Tidak bisa terhubung ke RAG API. Pastikan service berjalan.")
    except Exception as e:
        stop_event.set()
        await indicator_task
        logger.error(f"Error review skrip: {e}")
        await placeholder_msg.edit_text(f"⚠️ Terjadi error: {str(e)[:200]}")


# ── Handler: File upload (dokumen) ───────────────────────────
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle file upload — review skrip jika ekstensi sesuai."""
    doc = update.message.document
    user = update.effective_user
    filename = doc.file_name or "unknown"
    chat = update.message.chat

    # Cek ekstensi file
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SCRIPT_EXTENSIONS:
        supported = ", ".join(sorted(SCRIPT_EXTENSIONS))
        await update.message.reply_text(
            f"⚠️ File <b>{escape_html(filename)}</b> tidak didukung untuk review.\n\n"
            f"Ekstensi yang didukung: <code>{supported}</code>\n\n"
            "Atau paste isinya langsung di chat!",
            parse_mode="HTML",
        )
        return

    logger.info(f"File upload dari {user.first_name} (@{user.username}): {filename}")

    # Download file dari Telegram
    try:
        file_obj = await context.bot.get_file(doc.file_id)
        file_bytes = await file_obj.download_as_bytearray()
        script_content = file_bytes.decode("utf-8", errors="replace")
    except Exception as e:
        logger.error(f"Gagal download file: {e}")
        await update.message.reply_text(f"❌ Gagal membaca file: {str(e)[:200]}")
        return

    # Kirim ke review
    await _handle_script_review(update, chat, script_content, filename=filename)


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
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))

    # Jalankan bot (long polling)
    print("✅ Bot berjalan! Tekan Ctrl+C untuk berhenti.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

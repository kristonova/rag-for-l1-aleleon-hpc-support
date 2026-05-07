#!/usr/bin/env python3
"""
telegram_bot.py — Telegram Bot untuk RAG ALELEON HPC
Menerima pertanyaan dari user Telegram, kirim ke RAG API, lalu balas jawabannya.
Mendukung review skrip Bash/Slurm via paste teks atau file upload.
"""

import os
import asyncio
import logging
import re
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


def is_shell_script(text: str) -> bool:
    """Deteksi apakah teks yang di-paste adalah skrip Bash/Slurm.
    
    Hanya menganggap teks sebagai skrip jika:
    1. Dimulai dengan shebang bash/sh
    2. ATAU terdapat baris yang diawali dengan #SBATCH
    Ini mencegah salah deteksi jika user hanya menyebut '#SBATCH' di tengah kalimat.
    """
    text_stripped = text.strip()
    
    if text_stripped.startswith("#!/bin/bash") or \
       text_stripped.startswith("#!/bin/sh") or \
       text_stripped.startswith("#!/usr/bin/env bash") or \
       text_stripped.startswith("#!/usr/bin/env sh"):
        return True
        
    # Cek apakah ada baris yang diawali dengan #SBATCH (mengabaikan spasi di awal)
    if re.search(r'^\s*#SBATCH', text_stripped, flags=re.MULTILINE):
        return True
        
    return False


# ── Helper: convert Markdown ke HTML aman untuk Telegram ─────────
def markdown_to_telegram_html(text: str) -> str:
    """Mengubah sintaks Markdown dari LLM menjadi HTML yang valid untuk Telegram.
    
    Escape semua karakter khusus HTML (<, >, &) lebih dulu, lalu
    mengubah blok kode, bold, dan italic ke tag HTML yang diizinkan.
    """
    if not text:
        return ""

    # Escape HTML entities dasar (seperti <, >, &) agar tidak merusak parser Telegram
    text = html.escape(text, quote=False)

    # Fenced code blocks: ```lang\ncode\n```
    def replace_code_block(m):
        lang = m.group(1).strip()
        code = m.group(2)
        if lang:
            return f'<pre><code class="language-{lang}">{code}</code></pre>'
        return f'<pre><code>{code}</code></pre>'
    
    text = re.sub(r'```(\w*)\n?(.*?)```', replace_code_block, text, flags=re.DOTALL)

    # Inline code: `code`
    text = re.sub(r'`([^`\n]+)`', r'<code>\1</code>', text)

    # Bold: **bold** or __bold__
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text, flags=re.DOTALL)
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text, flags=re.DOTALL)

    # Italic: *italic* or _italic_
    text = re.sub(r'(?<!\w)\*(.+?)\*(?!\w)', r'<i>\1</i>', text, flags=re.DOTALL)
    text = re.sub(r'(?<!\w)_(.+?)_(?!\w)', r'<i>\1</i>', text, flags=re.DOTALL)

    # Headings: ### Title → <b>Title</b>
    text = re.sub(r'^#{1,6}\s+(.+)$', r'<b>\1</b>', text, flags=re.MULTILINE)

    # Strikethrough: ~~text~~ → <s>text</s>
    text = re.sub(r'~~(.+?)~~', r'<s>\1</s>', text, flags=re.DOTALL)

    # Links: [text](url) → <a href="url">text</a>
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)

    return text


def format_reply(answer: str, sources: list) -> str:
    """Format jawaban + sumber sebagai HTML."""
    reply = markdown_to_telegram_html(answer or "") + "\n"

    if sources:
        reply += "\n📚 <b>Sumber:</b>\n"
        for i, src in enumerate(sources, 1):
            title = html.escape(str(src.get("title") or "Unknown"), quote=False)
            section = html.escape(str(src.get("section") or ""), quote=False)
            url = src.get("source_url") or ""

            label = title
            if section:
                label += f" → {section}"

            if url:
                reply += f'{i}. <a href="{url}">{label}</a>\n'
            else:
                reply += f"{i}. {label}\n"

            justification = src.get("justification") or ""
            if justification:
                reply += f"   💡 <i>{html.escape(justification, quote=False)}</i>\n"

    return reply


def format_sources(sources: list) -> str:
    """Format hanya bagian sumber (untuk pesan terpisah)."""
    source_text = "📚 <b>Sumber:</b>\n"
    for i, src in enumerate(sources, 1):
        title = html.escape(str(src.get("title") or "Unknown"), quote=False)
        section = html.escape(str(src.get("section") or ""), quote=False)
        url = src.get("source_url") or ""

        label = title
        if section:
            label += f" → {section}"

        if url:
            source_text += f'{i}. <a href="{url}">{label}</a>\n'
        else:
            source_text += f"{i}. {label}\n"

        justification = src.get("justification") or ""
        if justification:
            source_text += f"   💡 <i>{html.escape(justification, quote=False)}</i>\n"

    return source_text


def format_review(review: str, issues_found: int, filename: str = None, policy_sources: list = None) -> str:
    """Format review skrip sebagai HTML yang aman."""
    header = "🔍 <b>Review Skrip"
    if filename:
        header += f": {html.escape(filename, quote=False)}"
    header += "</b>\n\n"

    result = header + markdown_to_telegram_html(review or "")

    if policy_sources:
        result += "\n\n📋 <b>Kebijakan HPC yang dirujuk:</b>\n"
        for i, src in enumerate(policy_sources, 1):
            title = html.escape(str(src.get("title") or "Unknown"), quote=False)
            section = html.escape(str(src.get("section") or ""), quote=False)
            url = src.get("source_url") or ""

            label = title
            if section:
                label += f" → {section}"

            if url:
                result += f'{i}. <a href="{url}">{label}</a>\n'
            else:
                result += f"{i}. {label}\n"

    return result


# ── Shared: Send reply ──────────────────────────────────────────
def split_html_for_telegram(text: str, max_len=4000) -> list:
    """Membagi teks HTML panjang menjadi potongan-potongan aman untuk Telegram.
    
    Menjaga agar tag HTML tidak terpotong di tengah dan otomatis
    menutup tag yang terbuka di akhir potongan, lalu membukanya 
    kembali di awal potongan berikutnya.
    """
    chunks = []
    while len(text) > max_len:
        # Cari titik pisah terbaik (paragraf atau baris baru)
        split_idx = text.rfind('\n\n', 0, max_len)
        if split_idx == -1:
            split_idx = text.rfind('\n', 0, max_len)
        if split_idx == -1:
            split_idx = max_len
            
        chunk = text[:split_idx]
        
        # Track tag HTML yang sedang terbuka
        open_tags = []
        for match in re.finditer(r'<(/)?([a-zA-Z0-9]+)([^>]*)>', chunk):
            is_close = match.group(1) == '/'
            tag_name = match.group(2)
            full_tag = match.group(0)
            
            if not is_close:
                open_tags.append((tag_name, full_tag))
            else:
                if open_tags and open_tags[-1][0] == tag_name:
                    open_tags.pop()
                    
        # Tutup tag yang masih terbuka di akhir chunk
        closing_tags = ''.join(f'</{tag_name}>' for tag_name, _ in reversed(open_tags))
        chunks.append(chunk + closing_tags)
        
        # Buka kembali tag tersebut di awal chunk berikutnya
        opening_tags = ''.join(full_tag for _, full_tag in open_tags)
        text = opening_tags + text[split_idx:].lstrip()
        
    if text.strip():
        chunks.append(text.strip())
        
    return chunks

async def _send_reply(placeholder_msg, update, reply_text: str):
    """Send message with HTML parse mode, fallback to plain text gracefully."""
    try:
        chunks = split_html_for_telegram(reply_text)
        
        # Kirim chunk pertama untuk me-replace placeholder
        await placeholder_msg.edit_text(chunks[0], parse_mode="HTML")
        
        # Kirim sisa chunk sebagai pesan baru
        for chunk in chunks[1:]:
            await update.message.reply_text(chunk, parse_mode="HTML")
            
    except Exception as e:
        logger.warning(f"Gagal kirim HTML ({e}), fallback ke teks murni tanpa tag.")
        # Hapus tag HTML jika parse gagal
        clean = re.sub(r'<[^>]+>', '', reply_text)
        clean = html.unescape(clean)
        
        clean_chunks = [clean[i:i+4000] for i in range(0, len(clean), 4000)]
        await placeholder_msg.edit_text(clean_chunks[0])
        for chunk in clean_chunks[1:]:
            await update.message.reply_text(chunk)


# ── Handler: /start ──────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Pesan selamat datang saat user pertama kali chat."""
    welcome = (
        "👋 Halo! Saya bot asisten HPC ALELEON.\n\n"
        "Saya bisa membantu dengan:\n"
        "📖 Pertanyaan — Kirim pertanyaan tentang ALELEON/Slurm/HPC\n"
        "📝 Review Skrip — Kirim/paste skrip .sh untuk saya review\n\n"
        "Contoh pertanyaan:\n"
        "• Bagaimana cara membuat conda environment?\n"
        "• Berapa kuota storage HOME untuk akun perseorangan?\n\n"
        "Review skrip:\n"
        "• Paste langsung skrip yang mengandung #!/bin/bash atau #SBATCH\n"
        "• Upload file .sh / .slurm / .sbatch\n\n"
        "Ketik pertanyaan atau kirim skrip! 🚀"
    )
    await update.message.reply_text(welcome)


# ── Handler: /help ───────────────────────────────────────────
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Tampilkan bantuan."""
    help_text = (
        "📖 Cara pakai bot ini:\n\n"
        "1. Pertanyaan:\n"
        "Ketik pertanyaan langsung di chat. Bot akan mencari jawaban dari wiki ALELEON.\n\n"
        "2. Review Skrip:\n"
        "• Paste skrip Bash/Slurm langsung di chat (harus mengandung #!/bin/bash atau #SBATCH)\n"
        "• Upload file .sh, .slurm, .sbatch, atau .bash\n"
        "Bot akan mengecek syntax, parameter SBATCH, dan best practice.\n\n"
        "Perintah:\n"
        "/start — Pesan selamat datang\n"
        "/help — Bantuan ini\n"
        "/status — Cek status RAG API"
    )
    await update.message.reply_text(help_text)


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

        reply = format_reply(answer, sources)
        await _send_reply(placeholder_msg, update, reply)

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
        logger.exception("Error detail:")
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
        policy_sources = data.get("policy_sources", None)

        reply = format_review(review, issues_found, filename=filename, policy_sources=policy_sources)
        await _send_reply(placeholder_msg, update, reply)

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
        logger.exception("Error review skrip:")
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
            f"⚠️ File *{filename}* tidak didukung untuk review.\n\n"
            f"Ekstensi yang didukung: `{supported}`\n\n"
            "Atau paste isinya langsung di chat!",
            parse_mode="Markdown",
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

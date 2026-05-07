"""
Script untuk melihat semua hasil parsing sitemap dalam bentuk tabel.
Jalankan: python debug_parse_sitemap.py
"""

import requests
import time
from xml.etree import ElementTree
from bs4 import BeautifulSoup
from langchain_core.documents import Document


def inspect_sitemap(sitemap_url, requests_per_second=2):
    # --- Parse sitemap ---
    print("Mengambil sitemap...")
    resp = requests.get(sitemap_url)
    root = ElementTree.fromstring(resp.content)
    ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = [loc.text for loc in root.findall(".//ns:loc", ns)]
    print(f"→ {len(urls)} URL ditemukan\n")

    # --- Tabel 1: Daftar semua URL ---
    print("=" * 100)
    print(f"{'No':<5} {'URL':<80} {'Status':<10}")
    print("=" * 100)

    page_results = []

    for i, url in enumerate(urls):
        try:
            time.sleep(1.0 / requests_per_second)
            page_resp = requests.get(url, timeout=30)
            status = page_resp.status_code
            soup = BeautifulSoup(page_resp.content, "lxml")
            content_div = soup.find("div", {"id": "mw-content-text"})
            has_content = "✓" if content_div else "✗ (no mw-content-text)"

            title_tag = soup.find("title")
            title = title_tag.text.strip() if title_tag else "(no title)"

            # Ambil plain text (1 artikel penuh, tanpa splitting)
            page_text = content_div.get_text(separator="\n", strip=True) if content_div else ""

            page_title = url.split("/wiki/")[-1].replace("_", " ") if "/wiki/" in url else url

            page_results.append({
                "no": i + 1,
                "url": url,
                "status": status,
                "has_content": content_div is not None,
                "title": title,
                "page_title": page_title,
                "content_text": page_text,
                "content_len": len(page_text),
            })

            print(f"{i+1:<5} {url:<80} {status} {has_content}")

        except Exception as e:
            page_results.append({
                "no": i + 1,
                "url": url,
                "status": "ERR",
                "has_content": False,
                "title": "",
                "page_title": "",
                "content_text": "",
                "content_len": 0,
            })
            print(f"{i+1:<5} {url:<80} ERROR: {e}")

    # --- Tabel 2: Ringkasan per halaman ---
    print("\n")
    print("=" * 120)
    print(f"{'No':<5} {'Title':<40} {'Text Length':<15} {'Has Content':<15} {'Status':<10}")
    print("=" * 120)
    for p in page_results:
        print(f"{p['no']:<5} {p['title'][:38]:<40} {p['content_len']:<15} {'✓' if p['has_content'] else '✗':<15} {p['status']:<10}")

    # --- Tabel 3: 1 Artikel = 1 Chunk (tanpa splitting) ---
    print("\n")
    print("=" * 140)
    print("HASIL PARSING (1 artikel = 1 chunk, tanpa splitting)")
    print("=" * 140)

    total_chunks = 0
    chunk_id = 0

    print(f"\n{'ChunkID':<10} {'Page Title':<40} {'Chars':<10} {'Preview':<80}")
    print("-" * 140)

    for p in page_results:
        if not p["has_content"] or not p["content_text"].strip():
            continue

        chunk_id += 1
        total_chunks += 1
        preview = p["content_text"][:77].replace("\n", " ") + "..."
        print(f"{chunk_id:<10} {p['page_title'][:38]:<40} {p['content_len']:<10} {preview:<80}")

    print("-" * 140)
    print(f"\nTOTAL: {total_chunks} chunks dari {len([p for p in page_results if p['has_content']])} halaman")

    # --- Tabel 4: Detail isi FULL setiap chunk ---
    print("\n")
    print("=" * 100)
    print("DETAIL ISI SETIAP CHUNK (1 artikel = 1 chunk, FULL tanpa potong)")
    print("=" * 100)

    chunk_id = 0
    for p in page_results:
        if not p["has_content"] or not p["content_text"].strip():
            continue

        chunk_id += 1
        print(f"\n┌─ Chunk {chunk_id} | Page: {p['page_title']}")
        print(f"│  Source: {p['url']}")
        print(f"│  Length: {p['content_len']} chars")
        print(f"├{'─' * 80}")
        # Tampilkan isi FULL tanpa potong
        for line in p["content_text"].split("\n"):
            print(f"│  {line}")
        print(f"└{'─' * 80}")

    print(f"\n{'='*60}")
    print(f"SELESAI. Total {total_chunks} chunks diinspeksi.")


if __name__ == "__main__":
    inspect_sitemap(
        sitemap_url="https://wiki.efisonlt.com/sitemap/sitemap-wiki.efisonlt.com-NS_0-0.xml",
        requests_per_second=2,
    )
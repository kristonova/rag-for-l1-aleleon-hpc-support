#!/usr/bin/env python3
"""
run-rag-bench.py — Benchmark Concurrency & Latency untuk RAG API
=================================================================
Mengirim pertanyaan nyata dari rag_app.py ke endpoint POST /ask
secara konkuren, lalu mengukur:
  • End-to-End Latency per request  (karena API non-streaming,
    ini mencakup retrieval + LLM generation)
  • P50 (median) & P99 latency
  • Requests Per Second (RPS)

Inspirasi: run-vllm-bench.sh — loop concurrency × input_length
           diganti menjadi loop concurrency × batch pertanyaan.

Dependensi: pip install aiohttp  (asyncio sudah bawaan Python)
=================================================================
"""

import asyncio
import aiohttp
import json
import os
import statistics
import time
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
# PERTANYAAN — diambil LANGSUNG dari rag_app.py pertanyaan_list
# ═══════════════════════════════════════════════════════════════
PERTANYAAN_LIST = [
    # LEVEL 1: Fakta Langsung / Direct Facts (20 pertanyaan)
    "Berapa kapasitas RAM efektif per node di partisi epyc-jumbo?",
    "Berapa batas maksimal walltime (waktu komputasi) per job untuk golongan akun perseorangan?",
    "Saya ingin pakai partisi GPU. GPU jenis apa yang terpasang di partisi ampere?",
    "Apa alamat website portal EFIRO Web Service (EWS) untuk login?",
    "Perintah apa yang harus saya ketik di terminal untuk melihat daftar environment python/pyload yang saya buat?",
    "Berapa harga 1 GPU Hour (GH) untuk pengguna golongan perseorangan non-akademia?",
    "Saya mau cek sisa kuota core hour saya. Perintah sausage apa yang harus diketik?",
    "Apa itu PKSPIAS dalam pendaftaran akun ALELEON?",
    "Jika saya pakai aplikasi SFTP seperti FileZilla, apakah ada limit ukuran file yang bisa diupload?",
    "OS (Sistem Operasi) apa yang digunakan oleh ALELEON Mk.V?",
    "Versi SLURM berapa yang terpasang di sistem ALELEON saat ini?",
    "Bagaimana cara membatalkan/menghentikan job yang berstatus PENDING di terminal?",
    "Berapa kapasitas limit storage HOME untuk akun perseorangan?",
    "Apakah sistem ALELEON memiliki backup jika saya tidak sengaja menghapus data di HOME?",
    "Email resmi apa yang harus saya hubungi jika ingin submit support ticket?",
    "Saya mau menjalankan simulasi GROMACS, apa nama binary MPI yang dipakai? Apakah gmx atau yang lain?",
    "Di EFIRO Account Manager, aplikasi authenticator apa saja yang didukung untuk fitur 2FA?",
    "Apa perintah terminal untuk mengecek status antrian job saya di Slurm?",
    "Apakah ALELEON mendukung instalasi package Python menggunakan pip?",
    "Modul Lmod apa yang harus saya load jika ingin menggunakan compiler GCC versi 15.2.0?",

    # LEVEL 2: Gabungan Info / Multi-Chunk (10 pertanyaan)
    "Saya ingin buka sesi interaktif JupyterLab menggunakan GPU. Apa bedanya partisi torti dan tilla, dan mana yang harus saya pilih?",
    "Saya punya file simulasi.ipynb. Bagaimana urutan langkah menjalankannya sebagai batch job di Job Composer EWS menggunakan conda environment saya sendiri?",
    "Jelaskan perbedaan arti status job 'PD' dan 'CG' saat saya mengecek squeue. Lalu sebutkan satu contoh Reason kenapa job bisa berstatus PD!",
    "Sebagai pengguna dari Akun Institusi, apakah job saya dibatasi maksimal 128 core CPU seperti akun perseorangan, dan apakah saya menggunakan sistem kuota (beli di awal)?",
    "Saya mau ganti password akun ALELEON saya. Di portal web mana saya harus login, dan menu apa yang harus diklik?",
    "File upload saya ukurannya 500 MB. Kenapa saya selalu gagal upload lewat menu Files di EWS, dan apa solusi spesifik serta alamat host yang harus saya gunakan?",
    "Saya ingin mengkompilasi code C++ menggunakan compiler AMD target Zen 2 dan OpenMPI terbaru. Modul apa saja yang harus saya module load secara berurutan?",
    "Saya mau pre-processing data GROMACS menggunakan binary gmx_mpi. Boleh tidak saya menjalankannya di Login Node? Jika boleh, apa syaratnya agar tidak di-kill admin?",
    "Jika saya menjalankan batch job lalu tiba-tiba koneksi internet rumah saya mati dan laptop saya disconnect dari VPN ALELEON, apakah job saya di Slurm ikut berhenti?",
    "Apa bedanya Effective Core Hour dengan Actual Core Hour di dalam sistem ALELEON?",

    # LEVEL 3: Reasoning / Deduksi & Troubleshooting (10 pertanyaan)
    "Saya menjalankan simulasi FLACS-CFD dengan 192 proses MPI murni. Di partisi epyc, otomatis job ini butuh lebih dari 1 node. Berapa angka yang harus saya tulis persisnya di #SBATCH --mem= jika total RAM yang saya butuhkan untuk seluruh job adalah 400GB?",
    "(Troubleshooting) Saya submit job GROMACS tapi selalu gagal dengan pesan error Invalid syntax. Di script saya menulis #SBATCH --mem= 64 GB. Apa yang salah dari tulisan saya?",
    "Saya submit 3 batch job berturut-turut. Job 1 pakai 64 CPU. Job 2 pakai 32 CPU. Kenapa saat saya submit Job 3 yang butuh 64 CPU, statusnya malah PENDING dengan tulisan QOSMaxCpuPerUserLimit, padahal node epyc masih banyak yang kosong?",
    "(Troubleshooting) Job saya berstatus PD dengan alasan AssocMaxWallDurationPerJobLimit. Di script saya menulis #SBATCH --time=4-00:00:00. Akun saya adalah akun perseorangan biasa. Mengapa tertahan?",
    "Saya mau menjalankan 10 simulasi FLACS-CFD sekaligus menggunakan fitur Slurm Array. Setiap simulasi butuh 4 core CPU dan 8GB RAM. Di script, apakah saya harus menulis --cpus-per-task=40 atau --cpus-per-task=4?",
    "Saya menjalankan script dengan #SBATCH --ntasks=4 dan #SBATCH --cpus-per-task=8 untuk OpenMX hibrida. Berapa total core thread CPU yang saya konsumsi, dan berapa Actual Core Hour yang terpotong jika job ini jalan 2 jam?",
    "Saya butuh komputasi memori raksasa sebesar 350 GB untuk satu aplikasi yang non-MPI (tidak bisa dibagi ke banyak node). Partisi apa yang WAJIB saya gunakan agar tidak error kehabisan memori?",
    "Saya mencoba mengisi form Sesi JupyterLab di EFIRO. Saya set waktu 3 hari (72 jam) dan minta 1 GPU. Namun tombol Launch ditolak karena saldo kurang. Jika sisa kuota GPU Hour (GH) saya tinggal 50 GH, berapa maksimal hari/jam yang bisa saya ajukan?",
    "Kenapa saat saya meminta alokasi #SBATCH --ntasks=7, sistem Slurm ALELEON akan membulatkannya menjadi 8 dan saya ditagih biaya untuk 8 core?",
    "Apakah ada gunanya saya upload file Slaster-Koster (SK) ke setiap ruang Job Composer DFTB+? Ataukah ada cara yang lebih hemat storage?",

    # LEVEL 4: Anti-Hallucination / Out-of-Context (15 pertanyaan)
    "Berapa kapasitas ukuran penyimpanan (storage) SSD untuk satu node Login di ALELEON?",
    "Bagaimana langkah-langkah submit job menggunakan aplikasi MATLAB di ALELEON?",
    "Berapa biaya denda yang harus dibayar jika file di HOME saya melebihi kuota 150GB?",
    "Apakah saya bisa menginstal package R menggunakan perintah conda install r-seurat di ALELEON?",
    "Berapa kecepatan internet/bandwidth VPN jika saya akses dari luar pulau Jawa?",
    "Bagaimana cara mereset environment Python bawaan sistem (python 3.9) ke kondisi pabrik jika saya merusaknya?",
    "Apakah tersedia modul aplikasi ANSYS Fluent di ALELEON?",
    "Bagaimana cara menyambungkan ekstensi Remote-SSH dari aplikasi Visual Studio Code (VSCode) ke compute node ALELEON?",
    "Saya adalah user dari Singapura (WNA). Berapa tarif konversi Core Hour ke dalam US Dollar (USD)?",
    "Apa password standar/bawaan dari admin sebelum saya menggantinya di awal?",
    "Siapa nama Chief Technology Officer (CTO) dari EFISON yang membangun ALELEON ini?",
    "Bagaimana cara menghapus halaman Wiki ALELEON jika saya menemukan typo?",
    "Bagaimana cara membatalkan/mengakhiri perjanjian PKSPIAS untuk akun Institusi sebelum waktunya habis?",
    "Jika server ALELEON mati lampu, berapa jam daya tahan baterai UPS yang dimiliki EFISON?",
    "Bagaimana cara menggunakan AutoGluon untuk machine learning di sistem ini?",

    # LEVEL 5: Pertanyaan Tambahan
    "Bagaimana cara membuat conda environment di aleleon?",
    "bagaimana cara menjalankan jupyter dengan conda environment sendiri?",
    "Versi Python default dari Anaconda3 2025.06-1 apa?",
    "Perintah apa untuk mengaktifkan Mamba 23.11.0-0?",
    "Bagaimana cara membuat modul pyload setelah conda env aktif?",
    "Perintah apa untuk melihat daftar modul pyload yang tersedia?",
    "Di partisi GPU mana batch job conda berjalan?",
    "Apa email support admin ALELEON?",
    "Jam kerja support EFISON kapan?",
    "Apa saja pilihan cara menjalankan komputasi Python dengan conda env di ALELEON?",
    "Apa perbedaan antara menjalankan batch job via Job Composer EWS dan via terminal Slurm?",
    "Bagaimana langkah lengkap membuat conda env baru dan modul pyload dari awal?",
    "Apa saja status job di squeue dan artinya masing-masing?",
    "Bagaimana cara mengisi formulir Jupyter di EWS untuk conda env user?",
]


def percentile(data: list, pct: float) -> float:
    """Hitung percentile tanpa numpy. pct dalam 0-100."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (pct / 100) * (len(sorted_data) - 1)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    d = k - f
    return sorted_data[f] + d * (sorted_data[c] - sorted_data[f])


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    question: str,
    request_id: int,
) -> dict:
    """
    Kirim 1 request POST /ask dan ukur latency.
    Return dict berisi metrik per-request.
    """
    payload = {"question": question}
    start_time = time.perf_counter()

    try:
        async with session.post(url, json=payload) as resp:
            status = resp.status
            body = await resp.json()
            end_time = time.perf_counter()
            latency = end_time - start_time

            return {
                "request_id": request_id,
                "question": question[:80],
                "status": status,
                "latency_s": round(latency, 4),
                "answer_length": len(body.get("answer", "")),
                "num_sources": len(body.get("sources", [])),
                "success": True,
            }
    except Exception as e:
        end_time = time.perf_counter()
        return {
            "request_id": request_id,
            "question": question[:80],
            "status": -1,
            "latency_s": round(end_time - start_time, 4),
            "answer_length": 0,
            "num_sources": 0,
            "success": False,
            "error": str(e),
        }


async def run_concurrency_level(
    url: str,
    questions: list,
    max_concurrency: int,
    num_requests: int,
) -> list:
    """
    Jalankan `num_requests` pertanyaan dengan max `max_concurrency`
    request paralel menggunakan asyncio.Semaphore.
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    # Timeout 10 menit per request (RAG bisa lambat)
    timeout = aiohttp.ClientTimeout(total=600)

    async def bounded_request(session, url, question, rid):
        async with semaphore:
            return await send_request(session, url, question, rid)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        for i in range(num_requests):
            q = questions[i % len(questions)]
            tasks.append(bounded_request(session, url, q, i + 1))

        results = await asyncio.gather(*tasks)
    return list(results)


def compute_summary(results: list, concurrency: int) -> dict:
    """Hitung summary statistik dari list hasil request."""
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    latencies = [r["latency_s"] for r in successful]

    if not latencies:
        return {
            "concurrency": concurrency,
            "total_requests": len(results),
            "successful": 0,
            "failed": len(failed),
            "error": "All requests failed",
        }

    total_time = max(r["latency_s"] for r in successful)  # wall-clock approx
    # More accurate: sum of all latencies / concurrency ≈ wall time
    # But for RPS we use actual wall time from first to last completion
    wall_start = 0  # all tasks start roughly together
    wall_time = total_time  # rough approximation

    summary = {
        "concurrency": concurrency,
        "total_requests": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "avg_latency_s": round(statistics.mean(latencies), 4),
        "min_latency_s": round(min(latencies), 4),
        "max_latency_s": round(max(latencies), 4),
        "p50_latency_s": round(percentile(latencies, 50), 4),
        "p99_latency_s": round(percentile(latencies, 99), 4),
        "stdev_latency_s": round(statistics.stdev(latencies), 4) if len(latencies) > 1 else 0,
        "throughput_rps": round(len(successful) / wall_time, 4) if wall_time > 0 else 0,
    }
    return summary


async def main():
    print("=" * 60)
    print("  RAG API Benchmark — Concurrency & Latency Test")
    print("=" * 60)
    print()

    # ── Konfigurasi: env var (container) atau interactive (lokal) ──
    # Jika RAG_API_URL di-set → mode non-interaktif (container)
    env_api_url = os.getenv("RAG_API_URL")
    is_container = env_api_url is not None

    if is_container:
        # Mode container: semua dari environment variables
        base_url = env_api_url.rstrip("/")
        num_requests = int(os.getenv("NUM_REQUESTS", str(len(PERTANYAAN_LIST))))
        print(f"  [Container Mode] RAG_API_URL = {base_url}")
    else:
        # Mode interaktif: prompt user
        host_input = input("Enter RAG API host (default: 0.0.0.0): ").strip()
        api_host = host_input if host_input else "0.0.0.0"

        port_input = input("Enter RAG API port (default: 8080): ").strip()
        api_port = port_input if port_input else "8080"

        num_input = input(f"Jumlah request per concurrency level (default: {len(PERTANYAAN_LIST)}): ").strip()
        num_requests = int(num_input) if num_input else len(PERTANYAAN_LIST)

        base_url = f"http://{api_host}:{api_port}"

    ask_url = f"{base_url}/ask"

    # Parse concurrency levels dari env (default: 1,2,4,8,16)
    conc_env = os.getenv("CONCURRENCY_LEVELS", "1,2,4,8,16")
    concurrency_levels = [int(x.strip()) for x in conc_env.split(",")]

    print()
    print(f"  Target URL        : {ask_url}")
    print(f"  Total pertanyaan  : {len(PERTANYAAN_LIST)}")
    print(f"  Request per level : {num_requests}")
    print(f"  Concurrency levels: {concurrency_levels}")
    print()

    # ── Health check (with retry for container startup ordering) ──
    max_health_retries = int(os.getenv("HEALTH_CHECK_RETRIES", "1" if not is_container else "30"))
    health_interval = int(os.getenv("HEALTH_CHECK_INTERVAL", "10"))

    print("Checking API health...")
    api_ready = False
    for attempt in range(1, max_health_retries + 1):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{base_url}/health", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    health = await resp.json()
                    status = health.get("status", "unknown")
                    print(f"  → Status: {status} (attempt {attempt}/{max_health_retries})")
                    if status == "ready":
                        api_ready = True
                        break
        except Exception as e:
            print(f"  ⚠️  Health check gagal: {e} (attempt {attempt}/{max_health_retries})")

        if attempt < max_health_retries:
            print(f"  ⏳ Retry dalam {health_interval}s ...")
            await asyncio.sleep(health_interval)

    if not api_ready:
        if is_container:
            print("  ❌ API tidak ready setelah semua retry. Keluar.")
            return
        else:
            proceed = input("  API belum ready. Lanjutkan? (y/n): ").strip().lower()
            if proceed != "y":
                print("Dibatalkan.")
                return
    print()

    # ── Result directory ──
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_base = os.getenv("RESULT_DIR", os.path.dirname(os.path.abspath(__file__)))
    result_dir = os.path.join(output_base, f"results-rag-{timestamp}")
    os.makedirs(result_dir, exist_ok=True)


    all_summaries = []


    for concurrency in concurrency_levels:
        print("=" * 60)
        print(f"  Concurrency: {concurrency}  |  Requests: {num_requests}")
        print("=" * 60)

        wall_start = time.perf_counter()
        results = await run_concurrency_level(
            ask_url, PERTANYAAN_LIST, concurrency, num_requests
        )
        wall_end = time.perf_counter()
        wall_elapsed = wall_end - wall_start

        summary = compute_summary(results, concurrency)
        # Override throughput with actual wall-clock time
        successful_count = summary["successful"]
        summary["wall_clock_s"] = round(wall_elapsed, 4)
        summary["throughput_rps"] = (
            round(successful_count / wall_elapsed, 4) if wall_elapsed > 0 else 0
        )
        all_summaries.append(summary)

        # Print summary
        print(f"  ✅ Successful : {summary['successful']}/{summary['total_requests']}")
        print(f"  ❌ Failed     : {summary['failed']}")
        if summary["successful"] > 0:
            print(f"  ⏱️  Avg latency : {summary['avg_latency_s']:.2f}s")
            print(f"  ⏱️  P50 latency : {summary['p50_latency_s']:.2f}s")
            print(f"  ⏱️  P99 latency : {summary['p99_latency_s']:.2f}s")
            print(f"  ⏱️  Min latency : {summary['min_latency_s']:.2f}s")
            print(f"  ⏱️  Max latency : {summary['max_latency_s']:.2f}s")
            print(f"  🕐 Wall clock  : {summary['wall_clock_s']:.2f}s")
            print(f"  🚀 Throughput  : {summary['throughput_rps']:.4f} req/s")
        print()

        # Save per-concurrency detail
        detail_file = os.path.join(
            result_dir, f"concurrency_{concurrency}_detail.json"
        )
        with open(detail_file, "w") as f:
            json.dump(
                {"summary": summary, "requests": results},
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"  💾 Detail saved: {detail_file}")
        print()

    # ── Save overall summary ──
    summary_file = os.path.join(result_dir, "benchmark_summary.json")
    with open(summary_file, "w") as f:
        json.dump(
            {
                "benchmark_info": {
                    "target_url": ask_url,
                    "total_questions": len(PERTANYAAN_LIST),
                    "requests_per_level": num_requests,
                    "concurrency_levels": concurrency_levels,
                    "timestamp": timestamp,
                },
                "results": all_summaries,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ── Print final table ──
    print()
    print("=" * 60)
    print("  BENCHMARK SUMMARY")
    print("=" * 60)
    header = f"{'Conc':>5} | {'OK':>4} | {'Fail':>4} | {'Avg(s)':>8} | {'P50(s)':>8} | {'P99(s)':>8} | {'RPS':>8}"
    print(header)
    print("-" * len(header))
    for s in all_summaries:
        print(
            f"{s['concurrency']:>5} | "
            f"{s['successful']:>4} | "
            f"{s['failed']:>4} | "
            f"{s.get('avg_latency_s', 0):>8.2f} | "
            f"{s.get('p50_latency_s', 0):>8.2f} | "
            f"{s.get('p99_latency_s', 0):>8.2f} | "
            f"{s.get('throughput_rps', 0):>8.4f}"
        )
    print()
    print(f"📁 All results saved to: {result_dir}")
    print(f"📊 Summary file: {summary_file}")


if __name__ == "__main__":
    asyncio.run(main())

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
# PERTANYAAN — diambil dari question.txt (49 pertanyaan, 4 level)
# ═══════════════════════════════════════════════════════════════
PERTANYAAN_LIST = [
    # =============================================================
    # LEVEL 1: Fakta Langsung / Direct Facts (12 pertanyaan)
    # Jawaban bisa ditemukan langsung di satu chunk/paragraf
    # =============================================================
    "Bagaimana perintah lengkap di terminal untuk menghapus sebuah conda environment beserta seluruh isinya di ALELEON?",
    "Arsitektur prosesor AMD generasi apa yang secara spesifik digunakan pada seluruh compute node CPU di ALELEON Supercomputer?",
    "Di direktori mana admin menyarankan pengguna untuk menyimpan folder file Slaster-Koster (SK) agar dapat diakses secara efisien oleh banyak job DFTB+?",
    "Apa perintah spesifik pyload yang harus saya jalankan di terminal untuk mengaktifkan environment ACPYPE versi 2023.10.27?",
    "Apa alamat tautan (URL) lengkap dari portal EFIRO Web Service (EWS) yang menjadi standar untuk mengakses fitur komputasi ALELEON?",
    "Siapa target spesifik kewarganegaraan dan tujuan dari diselenggarakannya program sponsorship layanan komputasi gratis EUREKA! oleh EFISON?",
    "Apa nama profil conda environment bawaan sistem atau perintah pyload yang harus diketikkan untuk memuat software BoltzTraP2 versi 25.3.1?",
    "Berapa jumlah maksimal job komputasi yang diizinkan untuk berstatus berjalan atau mengantri secara bersamaan bagi pengguna akun Uji Coba (Trial)?",
    "Sebutkan nama modul Lmod yang tepat untuk memuat NVIDIA HPC SDK Compilers versi 24.9!",
    'Berapa megabyte (MB) batas maksimal ukuran file yang dizinkan untuk diunggah langsung melalui menu "Edit Files" di dalam aplikasi Job Composer?',
    "Tombol menu apa yang harus saya klik secara presisi pada antarmuka aplikasi Job Composer di EFIRO Web Service jika saya ingin menghentikan job yang sedang berjalan?",
    'Apa kepanjangan dan arti dari kode status "CG" ketika saya mengecek antrian job komputasi saya menggunakan perintah squeue?',

    # =============================================================
    # LEVEL 2: Gabungan Info / Multi-Chunk (13 pertanyaan)
    # Butuh menggabungkan info dari beberapa bagian dokumen
    # =============================================================
    "Saya ingin menjalankan simulasi NAMD versi 2.14 khusus CPU murni menggunakan metode threading OpenMP. Modul Lmod apa yang harus saya muat, dan parameter SBATCH apa di dalam script yang harus saya set angkanya menjadi 1?",
    "Saya ingin membuka sesi JupyterLab dengan container PyTorch 2.3.0 berbasis ROCm 6.2 di partisi GPU trial. File .sif apa yang harus saya pilih di formulir, dan pada jenis GPU apa sesi saya akan berjalan?",
    "Jika saya ingin menggunakan modul cp2k/2025.2-gcc-13.4.0-cuda-12.9.0-k6i untuk komputasi, partisi apa yang WAJIB saya tuliskan di submit script, dan apakah modul tersebut mendukung eksekusi komputasi paralel multi-node?",
    "Job saya PENDING dengan alasan QOSMaxMemoryPerUserLimit. Apa definisi dari error tersebut, dan aplikasi beserta menu apa di terminal yang bisa saya gunakan untuk mengecek alokasi memori saya?",
    "Saya menggunakan template LAMMPS 22 July 2025 versi CPU dengan 150 proses MPI murni. Modul Lmod apa yang akan termuat di dalam script tersebut, dan secara default, ke berapa banyak node proses MPI saya tersebut akan disebar oleh Slurm jika berjalan di partisi epyc?",
    "Saya perlu mengunggah file input sebesar 2GB ke dalam ruang job Quantum ESPRESSO saya di Job Composer. Mengapa saya tidak bisa menggunakan fitur unggah bawaan EWS, dan di kolom manakah pada web EWS saya bisa menemukan alamat direktori spesifik untuk dituju oleh aplikasi SFTP saya?",
    "Saya ingin mengubah profil akun ALELEON Mk.V saya dan mengaktifkan 2FA menggunakan Google Authenticator. Di portal web beralamat apa saya harus login untuk melakukannya, dan menu apa yang harus saya tuju di dalamnya?",
    'Jika saya menutup secara paksa tab browser web saya saat sesi JupyterLab saya sedang berstatus "Running", apakah perhitungan Core Hour saya akan otomatis berhenti? Bagaimana cara yang paling benar untuk menghentikannya?',
    "Apa perbedaan mendasar cara kerja dan metode pencatatan antara 'Total Core Hour Usage' dengan 'Actual Core Hour Usage' yang ditampilkan di aplikasi sausage saat job komputasi saya sedang berlangsung?",
    "Saya ingin melakukan kompilasi software mandiri menggunakan CMake versi 3.31.8 yang didasari oleh compiler C/C++ GCC versi 14.2.0. Tuliskan dua perintah module load secara berurutan untuk memuat environment tersebut!",
    "Saya mensubmit job menggunakan modul openmpi/5.0.8-gcc-15.2.0-cuda-12.9.0-neq. Berapa kapasitas dan kecepatan maksimal interkoneksi jaringan (dalam Gbps) yang akan mendukung job saya jika ini dijalankan di partisi komputasi node GPU?",
    "Saya adalah pengguna akun golongan Perseorangan. Apa yang akan terjadi secara teknis pada akun saya jika masa aktif 1 tahun saya telah habis, dan berapa lama masa tenggang yang akan diberikan sistem sebelum akun saya sepenuhnya dinonaktifkan?",

    # =============================================================
    # LEVEL 3: Reasoning / Deduksi & Troubleshooting (12 pertanyaan)
    # Butuh menyimpulkan dari informasi yang tersedia
    # =============================================================
    "Saya memiliki saldo 100 CCH. Saya mensubmit job dengan alokasi 16 core CPU dan menulis parameter #SBATCH --time=10:00:00. Mengapa Slurm langsung menahan job saya dengan alasan AssocGrpCPUMinutesLimit padahal kondisi server (node) sedang kosong sama sekali dan jobnya belum berjalan satu detik pun?",
    "Saya mensubmit script simulasi dengan alokasi #SBATCH --nodes=2, #SBATCH --ntasks-per-node=4, dan #SBATCH --cpus-per-task=8. Jika job hibrida ini berhasil berjalan selama persis 2 jam, berapa total tagihan Actual Core Hour (CCH) yang secara matematis akan ditarik dari saldo saya?",
    "Saya ingin menjalankan simulasi dinamika molekuler NAMD berat yang dikonfigurasi untuk mendistribusikan proses ke 6 buah GPU NVIDIA RTX 3090 secara paralel. Berapa jumlah minimal node fisik di partisi ampere yang akan dikunci dan diokupasi oleh Slurm untuk memenuhi seluruh permintaan perangkat keras (GPU) job saya ini?",
    "Di dalam dokumen rujukan disebutkan bahwa GROMACS versi CUDA merupakan salah satu software yang dapat menyebarkan proses MPI secara langsung ke tingkatan perangkat keras GPU. Jika saya mensubmit GROMACS CUDA dan meminta 2 GPU, variabel environment internal apa dari Slurm yang bertugas menangkap jumlah alokasi proses MPI tersebut?",
    "Saya menjalankan simulasi CP2K dengan murni MPI dan secara spesifik mengatur nilai #SBATCH --ntasks=13. Jika job ini sukses berjalan selama 1 jam, apakah tagihan CCH akhir saya bernilai 13 CCH? Jelaskan analisis matematis Anda berdasarkan fitur SMT pada arsitektur perangkat keras prosesor AMD ALELEON!",
    "Saya mempunyai program Python custom (non-MPI) yang didesain untuk memuat dan mengolah sebuah dataset matriks raksasa berukuran 600 GB di dalam RAM. Saya merencanakan untuk mensubmit program ini ke partisi komputasi epyc-jumbo. Apakah job ini secara logis dan fisik bisa dieksekusi di ALELEON? Berikan alasan berbasis batasan maksimum memori efektif per node!",
    "Saya bermaksud mengeksekusi perintah gmx_mpi untuk melakukan pre-processing sistem molekul yang amat masif (yang memakan RAM 100GB dan komputasi 16 core CPU penuh) secara langsung di antarmuka terminal interaktif Aleleon Shell Access tanpa membuat submit script sbatch. Berdasarkan Standard Operating Procedure (SOP), apa risiko terbesar dan sanksi sistem yang akan dijatuhkan pada proses saya tersebut?",
    "Saya mencoba mengisi formulir sesi JupyterLab di EWS untuk partisi tilla dengan spesifikasi permintaan: 8 CPU thread, 1 GPU, dan durasi reservasi waktu selama 10 jam. Jika saldo yang saya miliki saat ini adalah 100 CCH dan 5 GH, mengapa tombol 'Launch' otomatis akan menggagalkan/menolak peluncuran sesi saya?",
    "Saya membuka aplikasi squeue dan melihat dua riwayat job historis saya. Job X tercatat berakhir dengan status 'CA' sedangkan Job Y berakhir dengan status 'PR'. Berdasarkan definisi status Slurm, siapa pihak yang secara teknis menjadi subjek/pelaku penghentian pada masing-masing job tersebut?",
    "Grup riset saya baru saja memborong pembelian kuota GPU Hour (GH) dalam jumlah besar khusus untuk mendongkrak performa dan kecepatan kecepatan simulasi ab-initio menggunakan software OpenMX 3.9. Mengapa investasi finansial (pembelian kuota GPU) ini justru berujung sama sekali tidak bisa dimanfaatkan secara teknis pada software tersebut?",
    "Sebagai pengguna akun golongan Perseorangan standar, saya mensubmit tiga batch job berskala berat di waktu yang bersamaan. Job A dialokasikan 64 core, Job B dialokasikan 64 core, dan Job C dialokasikan 16 core. Asumsikan klaster CPU epyc saat ini sedang tidak dipakai oleh siapa pun. Jika diproses oleh Slurm, manakah job yang akan langsung beralih ke status RUNNING dan manakah yang akan tertahan di status PENDING? Berikan analisis Anda!",
    "Dokumen menjelaskan bahwa Slurm ALELEON secara otomatis mendistribusikan proses MPI agar terkelompok ketat (tightly coupled) di dalam susunan grup CCX/CCD prosesor demi mengejar performa terbaik. Jika sebuah job dieksekusi dengan murni MPI yang meminta alokasi 32 proses, berapa banyak core fisik (physical cores) dan thread komputasi logis (v-cores) aktual yang akan dihidupkan/dialokasikan oleh sistem untuk melayani job tersebut?",

    # =============================================================
    # LEVEL 4: Anti-Hallucination / Strict Grounding (12 pertanyaan)
    # Jawaban TIDAK ada di dokumen, model harus jujur
    # =============================================================
    "Bagaimana contoh format pembuatan submit script lengkap beserta argumen module load untuk menjalankan simulasi pemodelan cuaca iklim resolusi tinggi menggunakan software WRF (Weather Research and Forecasting) di partisi komputasi epyc?",
    "Jika saya berstatus sebagai mahasiswa peneliti pascasarjana internasional yang berasal dan berafiliasi dengan universitas negeri di Malaysia, apakah saya secara sah berhak mendapatkan penerapan tarif diskon Golongan Perseorangan Akademia (sebesar Rp555/CCH)?",
    "Tolong sebutkan merk pabrikan komersial (vendor brand) spesifik dari kartu memori RAM dan papan induk (Motherboard) yang dipasang secara fisik di dalam kerangka sasis server pada compute node berpartisi ampere!",
    "Apa kombinasi syntax atau perintah bypass administrator khusus di dalam manajemen Slurm yang dapat saya ketikkan dari terminal pengguna (user) untuk melakukan restart (reboot) paksa dari jarak jauh pada salah satu compute node yang sedang mengalami hang atau tidak responsif?",
    "Berdasarkan catatan arsip historis di dokumen EUREKA! Periode 1 dan Periode 2, pada tanggal berapakah persisnya proses pengumuman pemenang untuk EUREKA! Periode 3 direncanakan akan dipublikasikan ke publik oleh EFISON?",
    "Saya memiliki skrip Python yang secara otomatis mengunduh dataset CIFAR-100 sebesar 16GB dari repositori cloud eksternal. Apakah compute node komputasi di partisi epyc dikonfigurasi untuk memiliki koneksi outbound internet publik secara langsung agar skrip saya ini tidak gagal saat dijalankan via sbatch?",
    "Berikan saya step-by-step tutorial mengenai bagaimana cara merutekan (forwarding) dan membuka antarmuka grafis (Graphical User Interface / GUI) visualizer bawaan dari software Quantum ESPRESSO langsung ke layar laptop Windows saya melalui koneksi terenkripsi SSH X11!",
    "Organisasi saya adalah pemegang Akun Institusi berbayar. Apakah kami secara perjanjian diizinkan untuk secara fisik membawa, menitipkan, dan menghubungkan rak perangkat penyimpanan eksternal pribadi (seperti SAN/NAS) milik institusi kami langsung ke dalam jaringan tertutup ALELEON?",
    "Berapa besar persentase nilai pengembalian dana (refund) tunai atau kompensasi kuota Core Hour yang secara resmi dijamin akan saya terima dari EFISON apabila seluruh infrastruktur sistem ALELEON mengalami kendala downtime (mati total) lebih dari 3x24 jam berturut-turut di tengah simulasi saya?",
    "Saya memiliki image kontainer Docker kustom yang mengharuskan eksekusi skrip entrypoint sebagai root. Apa kredensial kata sandi (password) sudo atau argumen sudo spesifik yang perlu saya tambahkan saat memanggil image tersebut dengan Apptainer di lingkungan komputasi ALELEON?",
    "Sesuai spesifikasi arsitektur kelas atas ALELEON, berapa nilai estimasi performa ukuran latensi jaringan (dalam satuan nanosecond / ns) yang diukur dari pengiriman message passing murni antar dua inti prosesor pada dua compute node epyc berbeda melewati switch Mellanox RoCE 100Gbps?",
    "Jika file proyek penelitian skripsi saya terhapus tanpa sengaja akibat salah ketik instruksi rm -rf, berapa lama sisa retention period maksimum (dalam hitungan batas hari) di mana file cadangan (backup snapshot) data saya tersebut masih aman disimpan oleh admin di sistem sebelum akhirnya tertimpa/terhapus permanen?",
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

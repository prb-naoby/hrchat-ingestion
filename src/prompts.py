"""
Centralized prompt builders for Gemini VLM.

Edit these helpers (or add new ones) to change how the VLM is instructed.
`vlm.py` imports and uses these so you don't need to touch the VLM client code.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


DEFAULT_SYSTEM_RULES = """\
Anda adalah asisten deskripsi dokumen dan gambar untuk kebutuhan kepatuhan/regulasi.
Tugas anda adalah memberikan penjelasan lengkap dan akurat berdasarkan apa yang TERLIHAT di gambar.
SELALU gunakan Bahasa Indonesia untuk seluruh keluaran.
Jika gambar berisi logo, sebutkan teks pada logo yang terlihat saja tanpa tambahan.
Jika gambar berisi diagram alur, berikan penjelasan per tahap.
Jika gambar berisi contoh draft dokumen, jelaskan elemen utamanya.
Fokus hanya pada informasi yang TERLIHAT / TERBACA di gambar atau halaman.
Jangan menebak atau menambah informasi yang tidak tampak.
Jika teks tidak terbaca, tulis "tidak terbaca" dan jangan menduga isinya.
Masukkan semua teks yang terlihat, tambahkan penjelasan singkat jika perlu.
Hormati format output yang diminta (plain text atau JSON) secara ketat.
Gunakan ejaan baku dan tanda baca yang baik.
Anda tidak perlu mendeskripsikan logo atau hal lainnya yang tidak berupa teks. Jika ada logo, cukup salin teks pada logo tersebut.
"""

TEXT_TASK_TEMPLATE = """\
TASK:
{task}

CONSTRAINTS:
- Bahasa: Indonesia (jawaban dalam Bahasa Indonesia meskipun teks/gambar berbahasa lain)
- Format: Plain text (ringkasan deskriptif)
- Cakupan konten:
  * Sebutkan judul/kepala dokumen jika terlihat.
  * Ringkas poin utama yang benar-benar terlihat/terbaca.
  * Jika ada tabel/angka/label yang jelas, sebutkan secara singkat.
  * Jika ada ketidakpastian, tulis "tidak yakin" (jelaskan bagian mana).
- Aturan ketat:
  * Jangan menambahkan pengetahuan di luar gambar.
  * Jangan menyimpulkan maksud/konteks di luar yang terlihat.
  * Jangan mengarang teks yang tidak terbaca.
  *  Anda tidak perlu mendeskripsikan logo atau hal lainnya yang tidak berupa teks. Jika ada logo, cukup salin teks pada logo tersebut
- Panjang: maksimum {max_words} kata
"""

JSON_TASK_TEMPLATE = """\
TASK:
{task}

CONSTRAINTS:
- Bahasa: Indonesia (jawaban dalam Bahasa Indonesia meskipun teks/gambar berbahasa lain)
- Format: JSON valid (tanpa backticks atau teks tambahan) dengan schema berikut:
  {{
    "judul": string atau null,
    "teks_terlihat": string atau null,         // kutipan singkat teks penting apa adanya (opsional)
    "poin_penting": [string],                  // ringkas poin yang benar-benar terlihat
    "label_dan_nilai": {{string: string}},     // pasangan label: nilai jika jelas (opsional)
    "ketidakpastian": string atau null         // jelaskan bagian "tidak yakin"/"tidak terbaca" bila ada
  }}
- Aturan ketat:
  * Hanya isi field jika informasinya benar-benar terlihat/terbaca.
  * Jika tidak ada informasinya, set ke null atau array kosong sesuai schema.
  * Jangan menambah field di luar schema di atas.
  * Jangan menambahkan pengetahuan eksternal.
- Panjang: maksimum {max_words} kata untuk gabungan ringkasan (bila relevan).
"""
# Catatan: Template di atas memaksa keluaran berbahasa Indonesia, menekankan "hanya yang terlihat",
# dan menyediakan slot untuk ketidakpastian, yang terbukti membantu menekan halusinasi.
# Rujukan: Prompt design strategies & structured output di Gemini API.

OCR_SYSTEM_RULES = """\
Anda adalah sistem OCR tingkat lanjut. Transkripsikan teks pada halaman secara teliti,
PERTAHANKAN tata letak dan struktur sebisa mungkin dengan Markdown (judul, daftar, tabel).
Jangan meringkas; salin teks apa adanya termasuk nomor, simbol, dan pemformatan garis baru.
Jika ada bagian yang tidak terbaca, tulis teks "-" di lokasi tersebut tanpa
menebak isinya. Gunakan Bahasa Indonesia hanya untuk catatan penanda seperti "-".
"""

OCR_TASK_TEMPLATE = """\
TASK:
{task}

PANDUAN:
- Kembalikan seluruh teks yang TERLIHAT pada halaman secara lengkap.
- Pertahankan urutan paragraf dan jeda baris seperti sumbernya.
- Gunakan Markdown untuk struktur:
  * Heading -> awali dengan tanda pagar (#) sesuai tingkat judul jika jelas.
  * Daftar bernomor atau bullet -> gunakan format Markdown (`1.` atau `-`).
  * Tabel -> gunakan tabel Markdown (baris header + pemisah + baris isi) bila memungkinkan.
- Jika karakter tidak dapat dikenali, gantikan dengan "-" tanpa menebak.
- Jangan menambahkan ringkasan, opini, atau informasi di luar teks yang terlihat.
- Baca ulang hasil untuk memperbaiki kesalahan ketik atau format.

Aturan Tambahan:
- Dokumen peraturan di Indonesia umumnya memiliki kop surat di bagian atas, abaikan bagian ini.
- Anda tidak perlu mendeskripsikan logo atau hal lainnya yang tidak berupa teks. Jika ada logo, cukup salin teks pada logo tersebut.
"""


# --- Public builders --------------------------------------------------------

def build_text_instruction(
    *,
    task: str,
    lang: str = "English",
    max_words: int = 120,
    system_rules: str = DEFAULT_SYSTEM_RULES,
) -> str:
    """Return the full instruction block for plain-text output."""
    # lang dipertahankan untuk kompatibilitas, tetapi kita menegaskan Bahasa Indonesia di template.
    return system_rules + "\n" + TEXT_TASK_TEMPLATE.format(
        task=task, lang=lang, max_words=max_words
    )


def build_json_instruction(
    *,
    task: str,
    lang: str = "English",
    max_words: int = 200,
    strict_json: bool = True,
    examples: Optional[List[Dict[str, Any]]] = None,
    system_rules: str = DEFAULT_SYSTEM_RULES,
) -> str:
    """Return the full instruction block for JSON output."""
    base = system_rules + "\n" + JSON_TASK_TEMPLATE.format(
        task=task, lang=lang, max_words=max_words
    )
    json_rule = (
        "Return ONLY valid JSON. Do not include backticks, code fences, or extra text."
        if strict_json
        else "Return JSON first; any comments after the JSON will be ignored."
    )
    example_block = ""
    if examples:
        # Use just the first example for brevity; extend as needed.
        from json import dumps
        example_block = "\nEXAMPLE JSON:\n" + dumps(examples[0], ensure_ascii=False, indent=2)

    return base + "\n" + json_rule + example_block


# --- Continuation builders (used when responses are truncated) --------------

def build_ocr_instruction(
    *,
    task: str,
    lang: str = "Indonesian",
    system_rules: str = OCR_SYSTEM_RULES,
) -> str:
    """Return instruction block for OCR-style full text transcription."""
    return system_rules + "\n" + OCR_TASK_TEMPLATE.format(task=task, lang=lang)


def continue_text_instruction(
    *,
    base_instruction: str,
    tail: str,
) -> str:
    """Instruction to continue a truncated plain-text answer."""
    return (
        f"{base_instruction}\n\n"
        "Lanjutkan PERSIS dari bagian terakhir. Jangan mengulang teks sebelumnya. "
        "Lanjutkan dari potongan berikut:\n<<<\n" + tail + "\n>>>"
    )


def continue_json_instruction(
    *,
    base_instruction: str,
    tail: str,
) -> str:
    """Instruction to continue a truncated JSON answer."""
    return (
        base_instruction
        + "\nLanjutkan OBJECT JSON YANG SAMA persis dari bagian terakhir "
          "(jangan ulangi key/value sebelumnya). "
          "Tambahkan hanya bagian yang kurang sehingga hasil akhir valid JSON.\n"
          "Konteks ekor:\n<<<\n"
        + tail
        + "\n>>>"
    )


# --- Back-compat convenience (optional) ------------------------------------

def default_prompt() -> str:
    """Legacy convenience function kept for compatibility."""
    return build_text_instruction(task="Jelaskan isi gambar secara ringkas.")


def ocr_prompt() -> str:
    """Prompt helper for OCR-style transcription."""
    return build_ocr_instruction(
        task=(
            "Transkripsikan seluruh teks pada halaman secara lengkap. "
            "Abaikan kop surat, logo, header, footer, nomor halaman, dan watermark; "
            "fokus hanya pada isi utama dokumen dan salin apa adanya dalam Markdown."
        )
    )

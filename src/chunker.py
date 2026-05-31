"""
Mecellem (Mursit-Large-TR-Retrieval) için Mevzuat Chunking Scripti
------------------------------------------------------------------
Strateji:
  - Her madde → 1 chunk (2048 token limitine sığıyorsa)
  - Limit aşan madde → N chunk'a bölünür, cümle sınırlarına göre
  - Her chunk'ın metadata'sında kanun + madde_no sabit kalır
  - Bölünen chunk'lara chunk_index ve total_chunks eklenir
  - RAG sırasında: chunk gelirse → aynı (kanun, madde_no) olanların
    tamamını getir → komple madde bağlamı sağlanır

Model Parametreleri (Mursit-Large-TR-Retrieval / ModernBERT-large):
  - Max sequence length : 2048 token
  - [CLS] + [SEP]       : 2 token overhead
  - Kullanılabilir       : 2046 token
  - Türkçe char/token   : ~3.5 (Türkçe özel vocab, 59,008 token)
  - Güvenli char limiti : 1800 token × 3.5 = 6,300 char (%12 buffer)
  - Bölme overlap       : ~50 char (cümle bütünlüğü için)
"""

import json
import re
from pathlib import Path


# ─── Sabitler ────────────────────────────────────────────────────────────────
SAFE_CHAR_LIMIT   = 6_300   # 1800 token × 3.5 char/token, %12 güvenlik buffer
MIN_CHUNK_CHARS   = 20      # Bunun altındaki maddeler skip edilir (boş/anlamsız)
INPUT_PATH        = "../MevzuatSources/mevzuat_temiz.json"
OUTPUT_PATH       = "../ChunkedFiles/mevzuat_chunked0.json"


# ─── Yardımcı Fonksiyonlar ───────────────────────────────────────────────────

def split_into_sentences(text: str) -> list[str]:
    """
    Türkçe hukuki metin için cümle bölücü.
    Parantez içindeki noktalara dikkat eder.
    """
    # Madde fıkra başlarını ayır: (1), (2), (a), (b) gibi
    text = re.sub(r'\s*\((\d+|[a-zğüşıöç])\)\s*', r'\n(\1) ', text)
    # Nokta + büyük harf veya yeni satır
    parts = re.split(r'(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ(])', text)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences


def split_text_to_chunks(text: str, limit: int = SAFE_CHAR_LIMIT) -> list[str]:
    """
    Metni cümle sınırlarını koruyarak limit'e sığan parçalara böler.
    Parçalar arasında overlap yoktur — hukuki metinde anlam kesintisi
    önlemek için cümle bütünlüğü korunur.
    """
    if len(text) <= limit:
        return [text]

    sentences = split_into_sentences(text)
    chunks   = []
    current  = []
    cur_len  = 0

    for sentence in sentences:
        s_len = len(sentence) + 1  # +1 boşluk
        if cur_len + s_len > limit and current:
            chunks.append(" ".join(current).strip())
            current = [sentence]
            cur_len = s_len
        else:
            current.append(sentence)
            cur_len += s_len

    if current:
        chunks.append(" ".join(current).strip())

    # Edge-case: tek bir cümle bile limiti geçiyorsa zorla böl
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= limit:
            final_chunks.append(chunk)
        else:
            # Kelime sınırında böl
            words = chunk.split()
            sub, sub_len = [], 0
            for word in words:
                w_len = len(word) + 1
                if sub_len + w_len > limit and sub:
                    final_chunks.append(" ".join(sub))
                    sub, sub_len = [word], w_len
                else:
                    sub.append(word)
                    sub_len += w_len
            if sub:
                final_chunks.append(" ".join(sub))

    return final_chunks


def make_chunk_id(kanun: str, madde_no: str, chunk_index: int) -> str:
    """
    Deterministik chunk ID: kanun + madde + index.
    Vector DB'de upsert/idempotency için kullanışlı.
    """
    base = f"{kanun}__{madde_no}__{chunk_index}"
    # Türkçe karakterleri normalize et, boşlukları alt çizgiye çevir
    base = re.sub(r'\s+', '_', base)
    base = re.sub(r'[^\w_-]', '', base)
    return base.lower()


# ─── Ana İşlem ───────────────────────────────────────────────────────────────

def process(input_path: str, output_path: str) -> dict:
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    output_chunks = []
    stats = {
        "toplam_madde"       : len(data),
        "skip_edilen"        : 0,
        "tek_chunk_madde"    : 0,
        "bolunmus_madde"     : 0,
        "toplam_chunk"       : 0,
        "bolunmus_detay"     : [],
    }

    for item in data:
        kanun       = item.get("kanun", "")
        madde_no    = item.get("madde_no", "")
        icerik      = item.get("icerik", "")
        encode_text = item.get("encode_text", "")
        sonraki     = item.get("sonraki_baslik", "")

        # Anlamsız kısa maddeler skip
        if len(encode_text.strip()) < MIN_CHUNK_CHARS:
            stats["skip_edilen"] += 1
            continue

        sub_chunks = split_text_to_chunks(encode_text, SAFE_CHAR_LIMIT)
        total      = len(sub_chunks)

        if total == 1:
            stats["tek_chunk_madde"] += 1
        else:
            stats["bolunmus_madde"] += 1
            stats["bolunmus_detay"].append({
                "kanun"    : kanun,
                "madde_no" : madde_no,
                "char_len" : len(encode_text),
                "n_chunks" : total,
            })

        for idx, chunk_text in enumerate(sub_chunks):
            chunk_obj = {
                # ── Embedding için gönderilecek metin ──
                "text": chunk_text,

                # ── Metadata (vector DB'ye indexlenecek) ──
                "metadata": {
                    "chunk_id"     : make_chunk_id(kanun, madde_no, idx),
                    "kanun"        : kanun,
                    "madde_no"     : madde_no,
                    "chunk_index"  : idx,
                    "total_chunks" : total,
                    "is_split"     : total > 1,
                    "sonraki_baslik": sonraki,
                    # Orijinal tam içerik: retrieval'da parent lookup için
                    "icerik_tam"   : icerik,
                },
            }
            output_chunks.append(chunk_obj)

        stats["toplam_chunk"] += total

    # Çıktıyı kaydet
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_chunks, f, ensure_ascii=False, indent=2)

    return stats, output_chunks


# ─── Çalıştır ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Chunking başlatılıyor...\n")
    stats, chunks = process(INPUT_PATH, OUTPUT_PATH)

    print("=" * 55)
    print("CHUNKING SONUÇLARI")
    print("=" * 55)
    print(f"  Toplam girdi madde    : {stats['toplam_madde']}")
    print(f"  Skip edilen (boş)     : {stats['skip_edilen']}")
    print(f"  Tek chunk madde       : {stats['tek_chunk_madde']}")
    print(f"  Bölünen madde         : {stats['bolunmus_madde']}")
    print(f"  ─────────────────────────────────")
    print(f"  Toplam çıktı chunk    : {stats['toplam_chunk']}")
    print()

    if stats["bolunmus_detay"]:
        print("BÖLÜNEN MADDELER:")
        for d in stats["bolunmus_detay"]:
            print(f"  [{d['n_chunks']} chunk] {d['kanun']} | {d['madde_no']} ({d['char_len']} char)")

    print(f"\nÇıktı kaydedildi → {OUTPUT_PATH}")

    # Örnek chunk göster
    print("\n" + "=" * 55)
    print("ÖRNEK CHUNK (ilk bölünen madde - chunk 0):")
    print("=" * 55)
    split_examples = [c for c in chunks if c["metadata"]["is_split"]]
    if split_examples:
        ex = split_examples[0]
        print(json.dumps(ex, ensure_ascii=False, indent=2)[:800])

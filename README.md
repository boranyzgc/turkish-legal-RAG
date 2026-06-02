# Demo
https://colab.research.google.com/drive/1GlxbI32wsaxBSS_kd9kXCW0I_JxqJx3e?usp=sharing


# Turkish Legal RAG System

A Retrieval-Augmented Generation (RAG) system for Turkish legal question answering, built with domain-specific fine-tuning at every pipeline stage.

## Covered Laws

| Law | No. |
|-----|-----|
| Türk Ceza Kanunu (TCK) | 5237 |
| Türk Medeni Kanunu (TMK) | 4721 |
| İş Kanunu | 4857 |
| Türk Borçlar Kanunu (TBK) | 6098 |
| Türkiye Cumhuriyeti Anayasası | 2709 |

## System Architecture

```
User Query
    │
    ▼
Fine-tuned Embedding Model
(newmindai/Mursit-Large-TR-Retrieval → mursit-large-tur2-v2)
    │  Dense retrieval, top-20 candidates
    ▼
Fine-tuned Cross-Encoder Reranker
(BAAI/bge-reranker-v2-m3 → bge-reranker-ft)
    │  Re-ranks to top-3
    ▼
Fine-tuned LLM
(Trendyol/Trendyol-LLM-8b-chat-v2.0 + LoRA adapter)
    │  Context-grounded answer generation
    ▼
Answer with Article Citations
```

## Results

Evaluated on a 161-question gold test set (human-annotated):

| System | Recall@5 | Recall@10 | MRR | nDCG@10 | Faithfulness | AnswerQuality |
|--------|----------|-----------|-----|---------|--------------|---------------|
| 1. Baseline RAG | 0.7341 | 0.8206 | 0.5752 | 0.6345 | 0.7519 | 0.5811 |
| 2. + Embedding FT | 0.8868 | 0.9329 | 0.7575 | 0.8005 | 0.7100 | 0.5984 |
| 3. + Reranker FT | 0.9281 | 0.9479 | 0.8457 | 0.8724 | 0.7702 | 0.5969 |
| 4. + LLM FT | 0.9281 | 0.9479 | 0.8457 | 0.8724 | 0.7739 | 0.5820 |
| **5. Full System** | **0.9281** | **0.9479** | **0.8457** | **0.8724** | **0.7581** | **0.5820** |

*Faithfulness and AnswerQuality evaluated using claude-sonnet-4-6 as LLM judge.*

## Repository Structure

```
turkish-legal-rag/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_data_preparation.ipynb       # Law corpus cleaning (TCK, TMK, İş K., TBK, Anayasa)
│   ├── 02_data_cleaning.ipynb        # Kaggle & HuggingFace QA dataset cleaning
│   ├── 03_embedding_finetuning.ipynb # Embedding model fine-tuning + synthetic data generation
│   ├── 04_reranker_finetuning.ipynb  # BGE cross-encoder reranker fine-tuning
│   ├── 05_llm_finetuning.ipynb       # LLM fine-tuning with QLoRA
│   ├── 06_evaluation.ipynb           # Ablation study & metrics
│   └── 07_demo.ipynb                 # Gradio interactive demo
├── src/
│   └── chunker.py                    # Mevzuat chunking script
└── data/
    ├── README.md
    ├── kaggle_train_clean.jsonl       # Cleaned Kaggle QA training set (11,488 samples)
    ├── kaggle_test_clean.jsonl        # Cleaned Kaggle QA test set (1,277 samples)
    ├── hf_train_clean.jsonl           # Cleaned HuggingFace QA training set (13,113 samples)
    ├── hf_test_clean.jsonl            # Cleaned HuggingFace QA test set (1,469 samples)
    ├── mevzuat_chunked0v2_normalized.json
    ├── embedding_train_synthetic.json
    ├── gold_test_matched_161_mastered_v02_filled.json└──
    ├── hard_negative_tur2.json
    └── data_README.md

```

> **Note:** Large files (mevzuat corpus, chunked index, gold test set, fine-tuned models) are hosted on HuggingFace Hub — see [Models & Data](#models--data) section.

## Notebooks

### 01 — Law Corpus Cleaning (`01_veri_hazirlama.ipynb`)
Cleans raw statute text from `muhammetakkurt/mevzuat-gov-dataset`:
- Per-statute cleaning functions (`tck_temizle`, `tmk_temizle`, `isk_temizle`, `tbk_temizle`, `anayasa_temizle`)
- Removes duplicate articles, fully-repealed (mülga) articles, and boilerplate entries
- Extracts and appends next-section titles to each article chunk
- Outputs `mevzuat_temiz.json`

### 02 — QA Dataset Cleaning (`02_data_cleaning.ipynb`)
Downloads and cleans two Turkish legal QA datasets:
- **Kaggle:** `batuhankalem/turkishlaw-dataset-for-llm-finetuning` (used for LLM fine-tuning — has context column)
- **HuggingFace:** general Turkish legal QA (used for supplementary training)
- Deduplication, normalization, train/test splitting

### 03 — Embedding Fine-tuning (`03_embedding_finetuning.ipynb`)
Fine-tunes `newmindai/Mursit-Large-TR-Retrieval` using iterative hard negative mining:
- Generates 4,648 synthetic (query, chunk) pairs from corpus using Claude Haiku API
- Two-round iterative hard negative mining (top-20 retrieval, 5 negatives per query)
- `CachedMultipleNegativesRankingLoss` (scale=20.0, mini_batch_size=32) — chosen over standard MNRL to avoid OOM on A100 with 1024-dim embeddings
- Training: 2 epochs, lr=5e-5, batch_size=32, fp16

### 04 — Reranker Fine-tuning (`04_reranker_finetuning.ipynb`)
Fine-tunes `BAAI/bge-reranker-v2-m3` cross-encoder:
- 27,888 training examples (4,648 queries × 1 positive + 5 hard negatives)
- Hard negatives mined using the fine-tuned embedding model
- Training: 3 epochs, lr=2e-5, batch_size=16, fp16
- Training loss: 0.34 → 0.09

### 05 — LLM Fine-tuning (`05_llm_finetuning.ipynb`)
Fine-tunes `Trendyol/Trendyol-LLM-8b-chat-v2.0` with QLoRA:
- Dataset: Kaggle only (11,488 examples) — chosen because it has a `context` column
- Context injected into the user turn to teach context-grounded generation
- LoRA: r=16, alpha=32, all attention + MLP projection layers, dropout=0.05
- Training: 2 epochs, lr=2e-4, batch_size=4 × grad_accum=8 (effective 32), bf16
- Saves only the LoRA adapter (~190MB)

### 06 — Evaluation (`06_evaluation.ipynb`)
Full ablation study across 5 system configurations:
- Retrieval metrics: Recall@5, Recall@10, MRR, nDCG@10
- Generation metrics: F1, ROUGE-L, AnswerQuality (LLM judge), Faithfulness (LLM judge)
- Hallucination rate = 1 − Faithfulness

### 07 — Gradio Demo (`07_demo.ipynb`)
Interactive Gradio interface with two modes:
- **Pre-loaded mode:** Query across the 5-statute mevzuat corpus
- **Custom document mode:** Upload any PDF/TXT file, auto-chunk and query

Models are downloaded from HuggingFace Hub at runtime (no local Drive required).

## Chunking (`src/chunker.py`)

Converts `mevzuat_temiz.json` → `mevzuat_chunked0.json`:
- One chunk per article if ≤ 6,300 characters (~1,800 tokens)
- Long articles split at sentence boundaries (no overlap — legal semantic coherence preserved)
- Each chunk includes `chunk_id`, `kanun`, `madde_no`, `chunk_index`, `total_chunks`
- Model limit: Mursit-Large max 2,048 tokens; safe char limit includes 12% buffer

## Models & Data

All fine-tuned models are hosted on HuggingFace Hub (private repos — request access from repo owner):

| Artifact | HuggingFace Hub |
|----------|----------------|
| Fine-tuned embedding model | `boranyzgc/mursit-large-tur2-v2` |
| Fine-tuned reranker | `boranyzgc/bge-reranker-ft` |
| Fine-tuned LLM adapter (LoRA) | `boranyzgc/trendyol-ft-context` |
| Chunked mevzuat corpus | `boranyzgc/turkish-legal-mevzuat` |

## Setup

### Requirements
```bash
pip install -r requirements.txt
```

### Running the Demo
Open `notebooks/07_demo.ipynb` in Google Colab (A100 GPU recommended):
1. Run all cells
2. Open the Gradio share link

### Reproducing Training
Each notebook is self-contained and runs on Google Colab Pro+ (A100 80GB).  
Mount Google Drive and update the `DATA_DIR` / `OUTPUT_DIR` paths at the top of each notebook.

Approximate GPU usage (A100, ~7.2 compute units/hour):

| Notebook | Duration | Compute Units |
|----------|----------|---------------|
| 03 Embedding FT (×2 rounds) | ~17 min | ~2 units |
| 04 Reranker FT | ~25 min | ~3 units |
| 05 LLM FT | ~90 min | ~11 units |
| 06 Evaluation | ~60 min | ~7 units |
| **Total** | **~3.2 hrs** | **~23 units** |

## Data

The `data/` folder contains cleaned QA datasets used for LLM fine-tuning:

| File | Source | Size | Used for |
|------|--------|------|----------|
| `kaggle_train_clean.jsonl` | Kaggle (`batuhankalem/turkishlaw-dataset-for-llm-finetuning`) | 11,488 | LLM FT training |
| `kaggle_test_clean.jsonl` | Kaggle | 1,277 | LLM FT validation |
| `hf_train_clean.jsonl` | HuggingFace Turkish Legal QA | 13,113 | Supplementary |
| `hf_test_clean.jsonl` | HuggingFace Turkish Legal QA | 1,469 | Supplementary |

Larger files (mevzuat corpus, chunked index, gold test set) are on HuggingFace Hub due to GitHub file size limits.

## Team

| Name | Contributions |
|------|--------------|
| Boran YÜZGEÇ | Law corpus cleaning, chunking, embedding FT, reranker FT, evaluation, Gradio demo |
| Ahmet Yiğit SALTEK | QA dataset cleaning, gold test set creation, chunking, metadata design, creating corpus structure, reranker FT, evalation |
| Başar GÖRGÜNDÜR | Huggingface-Kaggle data cleaning, vector embedding, Embedding model selection, LLM FT, evalatuion|

## Course

**CENG493 — Advanced Natural Language Processing**  
Çankaya University, 2025–2026 Spring

## License

MIT License — see [LICENSE](LICENSE) for details.

## Citation

If you use this work, please cite:

```bibtex
@misc{yuzgec2026turkishlegalrag,
  title  = {Turkish Legal RAG: Domain-Specific Retrieval-Augmented Generation for Turkish Statutory Law},
  author = {Yüzgeç, Boran and others},
  year   = {2026},
  url    = {https://github.com/boranyzgc/turkish-legal-rag}
}
```

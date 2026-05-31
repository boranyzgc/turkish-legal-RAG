# Data

This folder contains cleaned QA datasets used for LLM fine-tuning.

## Files

### `kaggle_train_clean.jsonl` — 11,488 samples
### `kaggle_test_clean.jsonl` — 1,277 samples
**Source:** [batuhankalem/turkishlaw-dataset-for-llm-finetuning](https://www.kaggle.com/datasets/batuhankalem/turkishlaw-dataset-for-llm-finetuning)  
**Used for:** LLM fine-tuning (train + validation)  
**Why Kaggle only:** This dataset includes a `context` column (retrieved law passage), which is required to teach the model context-grounded generation. The HuggingFace dataset lacks this column.

**Format:**
```json
{
  "soru": "Kasten öldürme suçunun cezası nedir?",
  "cevap": "TCK Madde 81 uyarınca müebbet hapis cezasıdır.",
  "context": "Türk Ceza Kanunu | MADDE 81\n\n(1) Bir insanı kasten öldüren kişi..."
}
```

---

### `hf_train_clean.jsonl` — 13,113 samples
### `hf_test_clean.jsonl` — 1,469 samples
**Source:** HuggingFace Turkish Legal QA dataset  
**Used for:** Supplementary reference (not used in final LLM fine-tuning — no context column)

**Format:**
```json
{
  "soru": "...",
  "cevap": "..."
}
```

---

## Files NOT in this repo (too large for GitHub)

The following files are hosted on HuggingFace Hub:

| File | Location | Description |
|------|----------|-------------|
| `mevzuat_temiz.json` | Drive / HF Hub | Cleaned statute corpus (5 laws, ~2300 articles) |
| `mevzuat_chunked0v2_normalized.json` | `boranyzgc/turkish-legal-mevzuat` | Chunked + normalized corpus (2,324 chunks) |
| `embedding_train_synthetic.json` | Drive | Synthetic QA pairs for embedding FT (4,648 samples) |
| `gold_test_normalized_matched_161.json` | Drive | Human-annotated gold test set (161 questions) |

## Cleaning Pipeline

See `notebooks/02_data_cleaning.ipynb` for the full cleaning pipeline:
- Downloaded from Kaggle API and HuggingFace
- Removed duplicates, empty answers, and malformed entries
- Normalized Turkish text (whitespace, encoding)
- 80/20 train/test split (stratified)

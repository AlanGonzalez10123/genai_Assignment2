# Assignment 2: Code Summarization via LSTM

**Course:** CSCI 455/555 — GenAI for Software Development, Spring 2026  
**Instructor:** Prof. Antonio Mastropaolo  

An encoder-decoder LSTM that generates natural language summaries for Java
methods, trained on ~50,000 code–docstring pairs mined from public GitHub
repositories and initialized with pretrained CodeT5+ embeddings.

---

## Data Sources and Pre-processing

### Sources
Public GitHub repositories filtered to Java, >100 stars, non-forked, fetched
via the GitHub Search API (top 700 by star count). 639 repositories were
successfully shallow-cloned.

### Mining Pipeline
1. For each cloned repo, up to 15 `.java` files are selected at random,
   excluding directories named `test`, `tests`, `example`, `examples`,
   `sample`, `demo`, or `generated`.
2. Each file is parsed with `javalang`. Every `MethodDeclaration` node that
   carries a Javadoc comment (`node.documentation`) is extracted together
   with its source text (recovered by brace-counting from the raw lines).
3. The Javadoc is cleaned: `/** */` delimiters and leading `*` characters are
   stripped, `@param`/`@return`/`@throws` tag lines are removed, and the
   remaining first-sentence text is lowercased and whitespace-normalised.
4. Method source is tokenized with `javalang.tokenizer` and flattened to a
   single whitespace-normalised line.

### Filtering
Each pair is kept only if it passes all of the following:
| Filter | Threshold |
|---|---|
| Non-ASCII characters in code | removed |
| Non-ASCII characters in summary | removed |
| Method token count | ≥ 5 |
| Boilerplate method name (`get*` / `set*` with < 20 tokens) | removed |
| Summary word count | ≥ 2 words |

### Deduplication and Split
Exact duplicates on tokenized code are removed. The remaining pairs are split
by repository rank (star-count order):

| Split | Repo ranks | Size |
|---|---|---|
| Train | 1 – 570 | ~50,000 |
| Validation | 571 – 600 | 1,000 (capped) |
| Test | 601 – 700 | ~7,000 (instructor set used for final eval) |

The split is by repository rather than by sample to prevent data leakage
between splits.

---

## Dependencies and Reproduction

### Requirements
- Python 3.10+
- CUDA-capable GPU recommended (tested on NVIDIA RTX 4050 with CUDA 11.8)
- Git (must be on PATH for repository cloning)

### Installation
```bash
# 1. Clone this repository and navigate into it
git clone <your-repo-url>
cd <repo-directory>

# 2. Place instructor-provided files in the repo root:
#      get_codet5_embeddings.py
#      requirements_side.txt
#      models/hard-negatives/   (SIDE model checkpoint directory)
#      dataset/sample_code.txt      (instructor test code)
#      dataset/sample_summary.txt   (instructor test summaries)
```

> **Note (Windows):** `sentencepiece==0.1.99` may fail to build from source.
> Install a pre-built wheel instead:
> `pip install sentencepiece --prefer-binary`

### Running the Notebook

Open `assignment-2-LSTM.ipynb` in Jupyter and run all cells top-to-bottom.
The notebook is fully self-contained and will:

1. **Cell 1** — install/verify all dependencies *(run once, then restart kernel)*
2. **Cell 2–3** — import libraries and set configuration constants
3. **Cells 4–9** — mine GitHub repos, extract pairs, filter, deduplicate, split,
   and write `.txt` files to `dataset/`
4. **Cell 10** — run `get_codet5_embeddings.py` to produce `.pt` embedding
   files (skipped automatically if outputs already exist)
5. **Cell 11–12** — load embeddings and build PyTorch `DataLoader`s
6. **Cell 13** — define the encoder-decoder LSTM model
7. **Cell 14** — define evaluation helpers (`compute_bleu1`, `ids_to_text`)
8. **Cell 15** — train with early stopping on validation BLEU-1 (patience = 3);
   best checkpoint saved to `checkpoints/lstm_codet5_summarization.pt`
9. **Cell 16** — plot training loss and validation BLEU-1 curves
10. **Cells 17–21** — load best checkpoint, run on test set, compute all
    metrics, display per-sample results, and save predictions
---

## Output Locations

| Output | Path |
|---|---|
| Train / val text files | `dataset/train_code.txt`, `dataset/train_summary.txt`, `dataset/val_code.txt`, `dataset/val_summary.txt` |
| Embedded token tensors | `dataset/train_code.pt`, `dataset/train_summary.pt`, `dataset/val_code.pt`, `dataset/val_summary.pt`, `dataset/test_code.pt`, `dataset/test_summary.pt` |
| Dataset metadata | `dataset/metadata.json` |
| Best model checkpoint | `checkpoints/lstm_codet5_summarization.pt` |
| Loss / BLEU curves | `checkpoints/loss_curve.png` |
| Test set predictions + metrics | `predictions/lstm_codet5_predictions.json` |

---

## Model Architecture

Encoder-decoder LSTM following the class notebook structure:

- **Embeddings:** pretrained CodeT5+ matrix (32,100 × 768), fine-tuned during training
- **Projection:** linear layer 768 → 256 applied to both encoder and decoder embeddings
- **Encoder:** 2-layer LSTM, hidden size 256, dropout 0.2
- **Decoder:** 2-layer LSTM, hidden size 256, dropout 0.2; teacher forcing during training, autoregressive at inference
- **Output head:** linear layer 256 → vocab size (32,100)
- **Training:** Adam (lr=1e-3), cross-entropy loss (padding ignored), gradient clipping at 1.0, early stopping on validation BLEU-1 with patience=3
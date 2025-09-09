# Confident Learning Label Auditor – Project Prompt

## Goal
Build a lightweight **Gradio app** that demonstrates *Confident Learning* for dataset label auditing.  
The app should:
- Take in a dataset (tabular, text, or image).
- Compute out-of-sample predicted probabilities.
- Run **cleanlab**’s Confident Learning routines to flag suspicious labels.
- Provide **engaging, interactive visualizations** for exploration.
- Allow **export** of a cleaned dataset and an audit report.

---

## Core Pipeline
1. **Data ingestion**
   - Tabular (CSV), text (CSV with `text` + `label`), image folders/ZIP.
2. **Model choice**
   - Tabular/Text → TF-IDF + Logistic Regression (+ calibration).
   - Image → Pretrained ResNet embeddings + Logistic Regression head.
3. **Predicted probabilities**
   - Use **K-fold cross-val** (`cross_val_predict` with `predict_proba`).
   - Ensure *out-of-fold* estimates (no leakage).
4. **Confident Learning**
   - Compute confident joint, label issues, quality scores, and suggested corrections.
5. **Outputs**
   - Ranked list of suspicious examples.
   - Suggested corrected labels.
   - Confidence/quality scores.

---

## Visualizations (must-have)
- **Confident Joint Heatmap**  
  Rows = given labels, cols = inferred labels.  
  Hover for counts, filter samples by clicking cells.
- **Uncertainty Scatter Plot**  
  x = self-confidence (prob of given label),  
  y = max alternative probability.  
  Points = samples; highlight suspicious clusters.

---

## Extra Visualizations (stretch)
- **Sample Inspector**: thumbnails/text snippets + approve/reject toggles.
- **Noise Simulation Slider**: inject 5–50% noise, show how visuals change.
- **Correction Replay Animation**: before/after relabel story.
- **Dataset Health Dashboard**: cards with noise rate, per-class trust, dataset score.

---

## Datasets to Use
- **Text**: AG News or 20-Newsgroups (subset).
- **Tabular**: UCI Adult (large) or Wine/Iris (small live demo).
- **Images**: CIFAR-10 (3–5 classes) or STL-10 subset.
- Inject 10–20% synthetic label noise for demo clarity.

---

## Persistence
- **MVP**: flat files + caching (`.npz` for embeddings/probs, `.parquet` for audit results).
- **Optional**: move to Postgres if multi-user with audit trails is required.
- **MongoDB** not needed unless JSON-first workflow is a priority.

---

## Project Structure

```

confident-learning-app/
├── app.py             # Gradio UI entrypoint
├── core.py            # Core audit logic (tabular/text/image)
├── models.py          # Embedding + classifier utilities
├── viz.py             # Visualization (heatmap, scatter, reports)
├── utils.py           # I/O helpers, noise injection, caching
│
├── data/              # Sample datasets (CSV, images, etc.)
├── cache/             # Precomputed embeddings / predicted probs
├── exports/           # Cleaned CSVs and generated reports
│
├── requirements.txt   # Python dependencies
└── README.md          # Documentation & demo instructions


```

 --- 



## Stretch Goals
- Add **downloadable HTML/PDF report** with visuals.
- Offer **“review mode”** where users can approve/reject relabel suggestions.
- Integrate with Hugging Face Spaces for public demo.

---

## Demo Script (3 min walk-through)
1. Upload dataset (CSV/ZIP).  
2. Run **Audit**.  
3. Show **Dataset Health cards**.  
4. Open **Heatmap** (point out confusing classes).  
5. Switch to **Scatter Plot** (show suspicious cluster).  
6. Inspect samples (thumbnails/text).  
7. Download **cleaned CSV** + **report**.  
8. (Optional) Turn on **Noise Simulation Slider** to dramatize effect.

---

## Key Packages
- `cleanlab`, `gradio`
- `scikit-learn`, `pandas`, `numpy`
- `matplotlib` or `altair/plotly` (for visuals)
- `torch`, `torchvision` (for embeddings, images)

---

**Next Step:** Start with tabular/text mode, build end-to-end pipeline (upload → audit → heatmap + table → export).  
Then extend to images and extra visualizations.


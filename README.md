# Human-Assisted Content Moderation (HITL Project)

Semester project for **Human-in-the-Loop Machine Learning (WS 26/27)**, Hochschule Bielefeld.

Online platforms receive more user comments than human moderators can review. A machine-learning model can flag potentially toxic comments, but difficult cases are often ambiguous, context-dependent, or judged differently by different people.

In this project we study how a toxicity classifier and human moderators can be combined effectively. The first deliverable focuses on **Active Learning**: *which comments should humans label so that the model learns with the least annotation effort?*

## Team

- Kavya Vadlakonda
- Ruthika Madhusudhana

## Deliverables

| Deliverable | Topic | Main notebook |
|---|---|---|
| 1 | Active Learning | [01_active_learning.ipynb](01_active_learning.ipynb) |

## Deliverable 1 – Active Learning

### Experimental setup

| Item | Setting |
|---|---|
| Dataset | Civil Comments (English) – 25,000 train / 5,000 validation / 5,000 test comments, sampled from the original partitions (seed 42) |
| Label | toxic = toxicity score ≥ 0.5 (~8% toxic) |
| Model | TF-IDF (1–2-grams, 20,000 features, sublinear tf) + Logistic Regression (`class_weight="balanced"`), as in the course baseline |
| Initial labeled set | 200 comments, stratified by label |
| Batch size / budget | 50 labels per round, 1,000 additional labels (20 rounds) |
| Repetitions | 5 seeds (42–46), each with a different initial labeled set |
| Evaluation | Validation macro-F1 (main metric) and toxic-class recall |
| Oracle | Reference labels, revealed only after a comment is selected (simulated annotation) |

### Acquisition strategies

- **Random sampling** – baseline for comparison
- **Uncertainty sampling** – least confidence (in binary classification, least confidence, margin and entropy give the same ranking)
- **Diversity sampling** – greedy farthest-first selection in TF-IDF space
- **Representative sampling** – relative representativeness (similar to the unlabeled pool, dissimilar to the labeled set)

### Main results (after 1,200 labels, mean ± std over 5 seeds)

| Strategy | Macro-F1 | Toxic recall | Toxic share in labeled set |
|---|---|---|---|
| **Uncertainty** | **0.641 ± 0.014** | **0.197** | **42.3%** |
| Diversity | 0.519 ± 0.008 | 0.041 | 6.3% |
| Representative | 0.518 ± 0.005 | 0.042 | 7.4% |
| Random | 0.512 ± 0.013 | 0.035 | 7.6% |

Reference: full training data (25,000 labels) = 0.707 macro-F1; majority baseline = 0.480.

**Key finding:** Uncertainty sampling clearly outperforms random sampling in all repetitions, reaching ~91% of the full-data performance with ~5% of the labels. It selects borderline comments, which are far more often toxic, and thereby re-balances the strongly imbalanced training data. Diversity and representative sampling (based on TF-IDF similarity) give no meaningful benefit over random sampling.

The full discussion and limitations are in the notebook.

## How to run

### Option A – Kaggle (used for our experiments)

1. Create a new notebook on [Kaggle](https://www.kaggle.com) and import `01_active_learning.ipynb` (**File → Import Notebook**).
2. In the session settings, switch **Internet** to **On** (required to download the dataset; Kaggle may ask for phone verification).
3. Click **Run All**. The full notebook runs in about 5–10 minutes on CPU; no GPU is needed.

### Option B – Local Jupyter

Requires Python 3.9 or newer.

```bash
git clone https://github.com/vadlakondakavya/content-moderation.git
cd content-moderation
pip install -r requirements.txt
jupyter notebook 01_active_learning.ipynb
```

### Option C – Course cluster without internet access

Download the dataset first with the course script from the [course repository](https://github.com/mbaezpy/hsbi-hitl-2026):

```bash
python scripts/download_dataset.py "google/civil_comments" "data/google_civil_comments"
```

Then, in the first code cell of the notebook, replace `load_dataset("google/civil_comments")` with `load_from_disk("<path>/data/google_civil_comments")` (import `load_from_disk` from `datasets`).

## Dataset

The notebook downloads the dataset automatically from Hugging Face: [`google/civil_comments`](https://huggingface.co/datasets/google/civil_comments). No data files are stored in this repository.

⚠️ *Content warning: the dataset contains offensive and discriminatory language.*
```

## References

- Course materials: [mbaezpy/hsbi-hitl-2026](https://github.com/mbaezpy/hsbi-hitl-2026) – project baseline and Labs 03–05
- Borkan et al. (2019). *Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification.* (Civil Comments dataset)
- Mathew et al. (2021). *HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection.*
- Bai et al. (2018). *RuG at GermEval: Detecting Offensive Speech in German Social Media.*
- Keller et al. *HOCON34k: A Corpus of Hate Speech in Online Comments from German Newspapers.*

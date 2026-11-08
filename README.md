# Valorant Rank Classification

Predicting a Valorant player's competitive rank from their in-game stats, using five classic ML classifiers,
with a Streamlit app to try the trained models on test data.

- **GitHub:** https://github.com/shubhamraj1601/Valorant-ML-Model
- **Live app:** https://valorant-ml-model-p84wtaa4ahawb3pcigejdx.streamlit.app/

## a. Problem statement

Given a player's lifetime competitive stats (damage per round, headshot %, K/D, win rate, most-played agents and
weapons, and so on), predict which rank they're in: **Immortal 1, Immortal 2, Immortal 3 or Radiant**.

It's a multi-class classification problem, and a harder one than it looks. Every player in the dataset is already
near the top of the ladder, so the gaps between neighbouring ranks are small and the classes are very imbalanced.

## b. Dataset description

**Source:** [Valorant Stats on Kaggle](https://www.kaggle.com/datasets/aliibrahim10/valorant-stats) (`val_stats.csv`, also included in this repo)

- **85,678 players, 38 columns**, from six regions (EU 32.7k, NA 20.9k, AP 19.3k, BR 7.0k, LAT 3.3k, KR 2.5k)
- **Target:** `rating`, the player's rank
- **Features used: 35** (28 numeric, 7 categorical)
  - Overall performance: `damage_round`, `headshots`, `headshot_percent`, `aces`, `clutches`, `flawless`, `first_bloods`,
    `kills`, `deaths`, `assists`, `kd_ratio`, `kills_round`, `most_kills`, `score_round`, `wins`, `win_percent`
  - Top 3 agents: `agent_1`, `agent_2`, `agent_3`
  - Top 3 weapons, each with name, head/body/leg hit % and kills: `gun1_*`, `gun2_*`, `gun3_*`
  - `region`
- `name` and `tag` were dropped, since they only identify the player

**Cleaning**

- Several numeric columns (`kills`, `deaths`, `headshots`, ...) were stored as text like `"1,506"`. These were converted to numbers.
- Removed 1,082 duplicate rows.
- The raw data has 19 rank values, but only four have a meaningful number of players. Everything below 100 players
  (Diamond, Platinum, Gold, Unrated, ...: 104 players in total) was dropped. That leaves 84,492 players:

  | Rank | Players | Share |
  |---|---|---|
  | Immortal 1 | 50,520 | 59.8% |
  | Immortal 2 | 20,530 | 24.3% |
  | Immortal 3 | 10,858 | 12.9% |
  | Radiant | 2,584 | 3.1% |

- Missing agent slots (some players only have one or two agents listed) are filled with the most common value.
  Numeric columns are scaled, and categorical ones are one-hot encoded.

**Split:** 80/20 stratified train/test split. To keep SVM training time reasonable, the models are trained on a
stratified sample of 20,000 players from the training split and tested on 5,000 from the test split. That test
set is [`test_data.csv`](test_data.csv), which is the file to upload in the app.

One thing to keep in mind when reading the results: a "model" that always guesses Immortal 1 would already be
about **60% accurate**. That's why MCC and AUC matter more than accuracy here.

## c. GitHub repository

https://github.com/shubhamraj1601/Valorant-ML-Model

```
├── app.py                  # Streamlit app
├── model_training.ipynb    # cleaning, training and evaluation of all five models
├── requirements.txt
├── README.md
├── test_data.csv           # held-out test set (5,000 players) for the app
├── val_stats.csv           # full dataset from Kaggle
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── support_vector_machine.pkl
    ├── label_encoder.pkl   # rank name <-> number
    ├── metadata.json       # feature list, target column, class names
    └── metrics.csv         # comparison table below
```

Each `.pkl` file is a full scikit-learn pipeline (imputing, scaling and one-hot encoding, then the model), so the
app can pass the raw rows from the CSV straight in.

## d. Models used

All five models were trained on the same data with the same preprocessing:

- **Logistic Regression** (`max_iter=2000`)
- **Decision Tree** (`max_depth=10`)
- **K-Nearest Neighbours** (`k=7`)
- **Gaussian Naive Bayes** (`var_smoothing=1e-2`, see the note below)
- **Support Vector Machine** (RBF kernel, `probability=True` so it can produce an AUC)

Because there are four classes, precision, recall and F1 are weighted by class size, and AUC is one-vs-rest.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.6106 | 0.7417 | 0.5369 | 0.6106 | 0.5444 | 0.2053 |
| Decision Tree | 0.5792 | 0.7012 | 0.5329 | 0.5792 | 0.5454 | 0.1757 |
| KNN | 0.5724 | 0.6615 | 0.5069 | 0.5724 | 0.5223 | 0.1309 |
| Naive Bayes | 0.5076 | 0.6846 | 0.5582 | 0.5076 | 0.5277 | 0.1996 |
| Support Vector Machine | 0.6058 | 0.7271 | 0.4871 | 0.6058 | 0.4921 | 0.1427 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | The best all-rounder: it has the highest accuracy, AUC and MCC. Stats like damage per round and win % shift fairly steadily as rank goes up, so a linear model captures most of the signal. It still leans heavily on Immortal 1, though: it finds 92% of Immortal 1 players but only 12–22% of the other ranks. |
| Decision Tree | Spreads its guesses across ranks more than Logistic Regression and does a bit better on Immortal 2 and 3, which gives it the best F1 by a hair. Its probabilities are coarse (one value per leaf), so its AUC is noticeably lower. The depth limit of 10 keeps it from memorising the training data. |
| KNN | The weakest of the "normal" models (lowest AUC and MCC). With dozens of one-hot agent and weapon columns, distances between players get noisy, and nearest neighbours are almost always Immortal 1 players, simply because there are so many of them. It predicted Radiant only 14 times out of 5,000. |
| Naive Bayes | Lowest accuracy, but second-best MCC and the best precision. It's the only model that actually tries to find Radiant players (44% recall on Radiant, against 14% or less for the others) instead of defaulting to Immortal 1. It assumes features are independent, which clearly isn't true here (kills, deaths, K/D and score are closely related), and that costs it overall accuracy. With the default smoothing it broke completely: the one-hot columns have almost no variance inside the small Radiant class, and it labelled 93% of players as Radiant (5% accuracy). Raising `var_smoothing` to 1e-2 fixed that. |
| Support Vector Machine | Its accuracy looks close to Logistic Regression's, but that's misleading: it predicts Immortal 1 for 93% of players and never correctly identifies a single Radiant player, which is why its MCC and F1 are low. It's also by far the most expensive model, taking about 13 minutes to train on 20k rows (everything else takes a few seconds) and the longest to make predictions. |
| **Overall winner** | **Logistic Regression.** It's best on accuracy, AUC and MCC, and it's also fast and tiny. That said, 61% accuracy against a 60% "always Immortal 1" baseline shows how hard it is to separate ranks this close together using lifetime stats alone. If the goal were specifically to spot Radiant players, Naive Bayes would be the better pick. |

## Streamlit app

The app ([`app.py`](app.py)) loads the saved models and scores an uploaded test file. It never retrains anything.

- **Upload a CSV** (or use the bundled `test_data.csv`, which can also be downloaded from the sidebar)
- **Pick a model** from the dropdown
- See all **six evaluation metrics**
- **Confusion matrix** and **classification report** for the chosen model
- Row-by-row predictions, downloadable as a CSV
- A side-by-side **comparison of all five models** on the uploaded file (switch it on at the bottom of the page)

SVM is slow at prediction time: scoring the 5,000-row test file takes a few minutes. Every other model takes a
second or two. Results are cached, so each model only runs once per file.

The uploaded CSV needs the 35 feature columns plus `rating`. Extra columns are ignored, and rows with a rank the
models weren't trained on are skipped with a warning.

## Running it locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (on macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

streamlit run app.py
```

To retrain the models, open `model_training.ipynb`, select the `.venv` kernel and run all cells.
It reads `val_stats.csv` from the repo by default (set `USE_KAGGLEHUB = True` to download it from Kaggle instead).
Expect around 15 minutes in total, almost all of it spent on SVM.

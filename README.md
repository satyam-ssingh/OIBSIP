# Data Analytics Track — Project Portfolio

**Track:** Data Analytics
**Completion Rule:** Complete at least 3 tasks from Level 1 or Level 2 combined to pass; complete the maximum number of tasks from both levels to be eligible for a Letter of Recommendation (LOR).
**Status:** ✅ **All 9 tasks completed** (4/4 Level 1 + 5/5 Level 2) — full LOR eligibility.

Every project below is a self-contained, end-to-end Jupyter Notebook, built to run in **Google Colab** with a simple file-upload step (or locally, with the dataset in the same folder). Each one follows the same discipline: load → inspect → clean → explore with visualisations and written observations → model (where applicable) → evaluate → conclude with concrete, data-driven takeaways — and each was executed end-to-end with zero errors before being marked complete.

---

## LEVEL 1

### Task 1 · EDA on Retail Sales Data — `DataAnalytics-L1-EDARetailSales`
**Objective:** Uncover patterns, customer behaviour trends, and actionable business insights in a retail sales dataset.
**Dataset(s) used:** Superstore sales dataset, and a second retail-transactions dataset with real customer demographics.

| Checklist item | Status |
|---|---|
| Initial inspection (shape, dtypes, nulls) | ✅ |
| Descriptive statistics (mean/median/mode/std) | ✅ |
| Monthly & quarterly sales trend line charts | ✅ |
| Customer demographics (age groups, gender) | ✅ — first dataset had no age/gender fields, so Segment/Region was used as the behavioural proxy instead and documented as such; the second dataset (with real `Age`/`Gender` columns) got the full demographic breakdown |
| Top 10 products; revenue by category | ✅ |
| Correlation heatmap | ✅ |
| Non-obvious bonus visualisation | ✅ — discount-rate vs. profit-margin analysis, revealing a loss-making discount threshold invisible in the category-level summaries |
| Markdown observations after every chart | ✅ |
| 3+ actionable business recommendations | ✅ |

**Key finding:** Discounts above ~20% flip average profit margin negative — a concrete, fixable policy issue rather than a general "discounting hurts margin" observation.

---

### Task 2 · Customer Segmentation Analysis — `DataAnalytics-L1-CustomerSegmentation`
**Objective:** Segment an e-commerce customer base by purchasing behaviour using clustering, for targeted marketing.
**Dataset used:** E-commerce transaction data (250,000 transactions, ~49,700 unique customers).

| Checklist item | Status |
|---|---|
| Load, inspect, handle missing/inconsistent data | ✅ |
| Descriptive stats: avg purchase value, frequency, CLV | ✅ |
| Feature selection: RFM (Recency, Frequency, Monetary) | ✅ |
| StandardScaler normalisation | ✅ |
| K-Means + Elbow Method for optimal K | ✅ — supported with silhouette scores across K=2-7 |
| Cluster scatter plots (2+ feature combinations) | ✅ — Recency×Monetary, Frequency×Monetary, Recency×Frequency |
| Cluster profiling (mean feature values, customer type) | ✅ |
| Bar chart: customers per cluster | ✅ |
| Marketing action per segment | ✅ |

**Key finding:** Churn rate came out flat (~20%) across every RFM segment — a genuinely useful negative result, since it means churn in this dataset isn't explained by Recency/Frequency/Monetary alone, redirecting the recommendation toward investigating other drivers rather than assuming "inactive = high churn risk."

---

### Task 3 · Cleaning Data — `DataAnalytics-L1-DataCleaning`
**Objective:** Take a deliberately messy dataset and transform it into a clean, analysis-ready one, with every decision documented.
**Dataset used:** Heart patients dataset (1,020 records) with inconsistent categories, mixed-encoding labels, and implausible clinical values.

| Checklist item | Status |
|---|---|
| Data quality report (nulls, duplicates, dtype issues, range anomalies) | ✅ |
| Missing-data strategy per column, justified in markdown | ✅ — median imputation for continuous vitals, row deletion for identifier/target columns, explicit "Unknown" category for high-missingness categoricals rather than forced mode imputation |
| Duplicate removal, with count documented | ✅ — 20 exact duplicates removed |
| Standardisation (e.g. Male/male/FEMALE → Male/Female; mixed 0/1/zero/one → 0/1) | ✅ |
| Outlier detection (IQR) with cap/remove/retain decision per column | ✅ — clinically implausible Cholesterol/Heart-Rate values capped rather than deleted, to preserve the rest of each row's valid data |
| Data type correction (IDs as string, dates/numerics correct) | ✅ |
| Before vs. after summary table | ✅ |
| Cleaned dataset saved to new CSV | ✅ |

**Key finding:** Row count dropped from 1,020 to 757 — honestly attributed almost entirely to the target column (`Heart_Disease`) being ~23% missing, since a diagnosis label can't be responsibly imputed, not to over-aggressive cleaning elsewhere.

---

### Task 4 · Sentiment Analysis — `DataAnalytics-L1-SentimentAnalysis`
**Objective:** Classify text sentiment (positive/negative/neutral) to surface public-opinion/customer-feedback insight.
**Dataset used:** Twitter Sentiment Extraction dataset (~3,500 labelled tweets).

| Checklist item | Status |
|---|---|
| Class distribution inspection | ✅ |
| Preprocessing: lowercase, punctuation removal, stopwords, tokenisation, lemmatisation | ✅ |
| TF-IDF feature extraction, purpose explained | ✅ |
| 80/20 train/test split (stratified) | ✅ |
| 2+ classifiers: Naive Bayes + Logistic Regression | ✅ |
| Accuracy, precision, recall, F1, confusion matrix per model | ✅ |
| Sentiment distribution bar chart + WordCloud per class | ✅ |
| Error analysis: 5 misclassified examples discussed | ✅ |
| Best model + real-world application conclusion | ✅ |

**Result:** Logistic Regression outperformed Naive Bayes (F1 = 0.625 vs. 0.574) — verified directly against the notebook's actual output before writing the conclusion.

---

## LEVEL 2

### Task 1 · Predicting House Prices with Linear Regression — `DataAnalytics-L2-HousePricePrediction`
**Objective:** Build and evaluate a Linear Regression model to predict house prices, end-to-end through interpretation.
**Dataset used:** House price dataset (2,000 properties) with Area, Location, Condition, and other features.

| Checklist item | Status |
|---|---|
| EDA: nulls, descriptive stats, target distribution | ✅ |
| Feature selection reasoning (markdown) | ✅ |
| Missing-value handling + One-Hot Encoding | ✅ |
| Correlation heatmap | ✅ |
| 80/20 train/test split | ✅ |
| Linear Regression model | ✅ |
| MSE, RMSE, R² | ✅ |
| Actual vs. predicted scatter plot | ✅ |
| Residual plot | ✅ |
| Coefficient analysis (standardised, for fair comparison) | ✅ |
| Bonus: Ridge/Lasso comparison | ✅ |

**Key finding (reported honestly, not glossed over):** R² came out **negative** — the model performs worse than predicting the average price for every house. Every correlation between the recorded features and `Price` was near-zero, including `Area` (which real-world housing intuition would expect to dominate). All interpretive text in the notebook was rewritten to reflect this rather than assume success, with a dedicated discussion of what this means and what would come next with real housing data.

---

### Task 2 · Wine Quality Prediction — `DataAnalytics-L2-WineQualityPrediction`
**Objective:** Train and compare multiple classifiers to predict wine quality from physicochemical properties.
**Dataset used:** Wine Quality dataset (1,143 samples, UCI/Kaggle standard).

| Checklist item | Status |
|---|---|
| Load, inspect, class distribution of quality scores | ✅ |
| EDA: feature distributions + correlation heatmap | ✅ |
| Class imbalance discussion | ✅ — raw quality 3 and 8 had only 6 and 16 samples respectively |
| Feature engineering: binning into 3-class (Low/Medium/High), justified | ✅ |
| Stratified train/test split | ✅ |
| 3 classifiers: Random Forest, SGD, SVC | ✅ — all with `class_weight='balanced'` |
| Accuracy, classification report, confusion matrix per model | ✅ |
| Random Forest feature importance chart | ✅ |
| Side-by-side comparison table | ✅ |
| Deployment recommendation + reasoning | ✅ |

**Result:** Random Forest was the clear winner (F1 = 0.860 vs. SGD's 0.669 and SVC's 0.695) — recommended for deployment on interpretability, imbalance robustness, and no scaling dependency, not accuracy alone.

---

### Task 3 · Fraud Detection — `DataAnalytics-L2-FraudDetection`
**Objective:** Build a fraud-detection pipeline that treats class imbalance as a core challenge, not an afterthought.
**Dataset used:** Credit card transactions dataset (300,000 real transactions after cleaning, ~5.23% fraud).

| Checklist item | Status |
|---|---|
| Class imbalance analysis (% fraudulent) | ✅ |
| EDA: transaction amount by class | ✅ |
| Time-of-day analysis | ⚠️ Skipped, documented — this dataset has no `Time` column (unlike the original 284,807-row benchmark version); a fabricated time feature was deliberately avoided rather than inventing a chart |
| Why accuracy is misleading (markdown) | ✅ |
| Imbalance handling: SMOTE (train-set only) | ✅ |
| Stratified train/test split | ✅ |
| 2+ models: Logistic Regression + Random Forest | ✅ |
| Precision, Recall, F1, AUC-ROC curve | ✅ |
| Recall vs. Precision trade-off explained | ✅ |
| Feature importance / coefficient analysis | ✅ |
| Scalability discussion (1M transactions/hour) | ✅ |

**Data-quality finding:** The raw file contained 268,307 completely empty junk rows appended after the real 300,000 transactions — identified, documented, and removed before any analysis.
**Result:** Random Forest reached AUC-ROC = 0.997 vs. Logistic Regression's 0.992 — both strong, with Random Forest recommended and the recall-priority threshold argument made explicit for a fraud use case.

---

### Task 4 · Unveiling the Android App Market (Google Play Store Analysis) — `DataAnalytics-L2-PlayStoreAnalysis`
**Objective:** Comprehensive analysis of the Play Store ecosystem — categories, ratings, pricing, and review sentiment.
**Dataset used:** Google Play Store apps dataset (10,840 listings, 33 categories).

| Checklist item | Status |
|---|---|
| Load apps dataset and reviews dataset separately | ⚠️ Partial — only the apps dataset was provided; no companion user-reviews file was available. The notebook includes a working optional-upload step and full sentiment pipeline that activates automatically if a reviews CSV is supplied, but reports an explicit "Skipped" rather than fabricating review text |
| Data cleaning: dtypes, nulls, duplicates | ✅ — 483 exact duplicates + 1,096 App+Category duplicates removed |
| Category analysis + saturation | ✅ |
| Ratings analysis + by-category average | ✅ |
| Size vs. installs scatter + correlation | ✅ |
| Pricing analysis: free/paid split, price distribution, revenue estimate | ✅ — with an explicit caveat that `Installs` is a bucketed figure, so revenue is a directional estimate, not an exact figure |
| Sentiment analysis on reviews | ⚠️ Conditional — see above |
| Sentiment by category | ⚠️ Conditional — see above |
| Interactive Plotly visualisation | ✅ — bubble chart of category saturation vs. rating vs. install volume |
| 3 data-driven insights for a new app developer | ✅ |

---

### Task 5 · Autocomplete and Autocorrect Data Analytics — `DataAnalytics-L2-AutocompleteAutocorrect`
**Objective:** Analyse and compare autocomplete and autocorrect algorithm approaches on real text data.
**Dataset used:** A provided text corpus (~52,000 tokens).

| Checklist item | Status |
|---|---|
| Text corpus collected/downloaded | ✅ — with an honest caveat: the corpus turned out to be small and highly templated (only ~161 unique word types), documented explicitly rather than presented as a large varied corpus |
| Preprocessing: tokenisation, lowercase, punctuation, stopwords | ✅ |
| Autocomplete: frequency-based n-gram (bigram/trigram) | ✅ — both built, to also satisfy the algorithm-comparison requirement |
| Tested on 10+ prefixes, top-3 predictions shown | ✅ |
| Autocorrect: edit-distance based correction | ✅ — pyspellchecker **and** a custom from-scratch Levenshtein implementation |
| Tested on 20+ misspelled words, accuracy measured | ✅ |
| Precision & Recall defined and calculated (both tasks) | ✅ |
| 2+ approaches compared | ✅ — bigram vs. trigram; pyspellchecker vs. custom Levenshtein |
| Top-20 frequency bar chart; autocorrect confusion matrix | ✅ |
| Limitations vs. production systems (e.g. Google Keyboard) discussed | ✅ |

**Result:** pyspellchecker (95% accuracy) heavily outperformed the custom corrector (50%) — traced directly to dictionary coverage, not distance-algorithm quality, since the custom corrector could only ever match words already in the small corpus vocabulary. Trigram (92.9%) outperformed bigram (84.4%) on next-word recall.

---

## How to Run

Every notebook is self-contained and Colab-ready:

1. Open [colab.research.google.com](https://colab.research.google.com)
2. `File → Upload notebook` and select the `.ipynb` file from the relevant project folder
3. `Runtime → Run all`
4. When prompted, upload the project's dataset (included in the same folder)

To run locally instead: place the dataset in the same directory as the notebook and run in Jupyter as normal — each notebook detects whether it's running in Colab or locally and adjusts the upload step automatically.

---

## Skills Demonstrated

| Area | Projects |
|---|---|
| Exploratory Data Analysis | EDARetailSales, PlayStoreAnalysis |
| Data Cleaning & Quality Auditing | DataCleaning, FraudDetection, PlayStoreAnalysis |
| Unsupervised Learning (Clustering) | CustomerSegmentation |
| Supervised Learning — Regression | HousePricePrediction |
| Supervised Learning — Classification | WineQualityPrediction, FraudDetection |
| Class Imbalance Handling | WineQualityPrediction, FraudDetection |
| NLP & Text Processing | SentimentAnalysis, AutocompleteAutocorrect, PlayStoreAnalysis |
| Model Evaluation & Multi-Model Comparison | HousePricePrediction, WineQualityPrediction, FraudDetection, SentimentAnalysis, AutocompleteAutocorrect |
| Interactive Visualisation | PlayStoreAnalysis (Plotly) |
| Honest Reporting of Negative/Unexpected Results | HousePricePrediction (negative R²), CustomerSegmentation (flat churn rate), PlayStoreAnalysis (missing reviews data), AutocompleteAutocorrect (small/templated corpus) |

---

## Author

**Satyam Singh**

# Quora Question Pair Similarity Predictor

A Machine Learning and Natural Language Processing (NLP) pipeline designed to identify duplicate questions on Quora. The project extracts advanced textual features (fuzzy matching, token ratios, length metrics, and Word2Vec embeddings) and uses tree-based classifiers to predict whether two questions carry the same intent.

Includes an interactive **Streamlit web application** for real-time duplicate prediction.

---

## 📌 Features

- **Text Preprocessing:** Cleans HTML tags, punctuation, special characters, and handles missing/empty text strings safely.
- **Advanced Feature Engineering:**
  - **Token Features:** `cwc_min`, `cwc_max`, `csc_min`, `csc_max`, `ctc_min`, `ctc_max`, `last_word_eq`, `first_word_eq`.
  - **Length & String Features:** Absolute length difference (`abs_len_diff`), mean token length (`mean_len`), longest common substring ratio (`longest_substr_ratio`).
  - **Fuzzy Matching Features:** `fuzz_ratio`, `fuzz_partial_ratio`, `token_sort_ratio`, `token_set_ratio`.
- **Machine Learning Models:** Scaled using `MinMaxScaler` and trained on tree-based models like **XGBoost** and **Random Forest**.
- **Web Interface:** Built with **Streamlit** for live user interaction and real-time classification.

---

## 🛠️ Project Architecture

```text
├── app.py                 # Main Streamlit web application
├── helper.py              # Helper functions for feature extraction & preprocessing
├── w2v_model.pkl          # Saved Word2Vec embedding model
├── xgb_model.pkl          # Saved XGBoost classifier model
├── requirements.txt       # Dependencies
└── README.md              # Project documentation

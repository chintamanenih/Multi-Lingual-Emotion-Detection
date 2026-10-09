# Multi-Lingual Emotion Detection Engine

> An advanced, multi-modal NLP ensemble framework designed to detect nuanced human emotions across English, Hindi, and Spanish text by combining custom language-specific preprocessing, Google Translation alignment, and weighted multi-BERT architectures.

---

## 🏗️ Architecture & Pipeline Overview

* **Advanced Text Preprocessing**: Custom tokenization, stemming, and lemmatization workflows tailored for English, Spanish, and a custom-built dictionary/stemming module specifically engineered for Hindi text.
* **Translation Alignment**: Integrates Google Translation pipelines to normalize foreign inputs, allowing cross-lingual inference through unified English models.
* **Ensemble Multi-Model Architecture**: Concurrently executes language-agnostic multilingual BERT, translated-data BERT, and language-specific BERT models (34 models simultaneously), aggregating predictions through a weighted scoring mechanism to maximize accuracy.

---

## 🚀 Tech Stack
* **Machine Learning & NLP**: PyTorch, Hugging Face Transformers, BERT, DistilBERT, Scikit-learn
* **Preprocessing & Translation**: Pandas, NumPy, Google Translate API, Custom Rule-Based Dictionaries
* **Application**: FastAPI, Streamlit (pp.py, Translated_app.py)
* **DevOps**: Git, GitHub Actions

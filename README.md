# 📝 Resume Matching with NLP and Sentence Embeddings

This project uses Natural Language Processing (NLP) techniques to automatically match candidate resumes to job descriptions.  
It applies text vectorization (TF-IDF and sentence-transformers) and cosine similarity to find the best matches.

Includes a Streamlit web application for uploading resumes and viewing top job matches.

## 📂 Project Structure
- `data/` - Raw and processed resumes and job descriptions
- `notebooks/` - Jupyter notebooks for EDA and modeling
- `src/` - Modular Python scripts for text processing and model building
- `app/` - Streamlit web app for user interaction

## 🚀 How to Run
1. Install requirements:
```bash
pip install -r requirements.txt
```
2. Launch Jupyter and run notebooks:
```bash
jupyter notebook notebooks/1_EDA.ipynb
jupyter notebook notebooks/2_Modeling.ipynb
```
3. Start the Streamlit app:
```bash
streamlit run app/streamlit_app.py
```

## 🛠 Technologies Used
- scikit-learn
- nltk
- sentence-transformers
- streamlit
- pandas, numpy

## 🔥 Future Work
- Fine-tune transformer models on resume-job pairs
- Implement Named Entity Recognition (NER) for skill extraction
- Add PDF resume parsing support (currently plain text)
- Deploy app with Docker and cloud hosting (AWS/Azure)

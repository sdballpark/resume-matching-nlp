import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title('📝 Resume Matcher Demo')

uploaded_resume = st.text_area("Paste your resume text here:")

if uploaded_resume:
    jobs = pd.read_csv('data/raw/jobs.csv')

    # Vectorize
    vectorizer = TfidfVectorizer()
    job_vecs = vectorizer.fit_transform(jobs['job_description'])
    resume_vec = vectorizer.transform([uploaded_resume])

    # Similarity
    similarities = cosine_similarity(resume_vec, job_vecs).flatten()

    jobs['similarity_score'] = similarities
    top_jobs = jobs.sort_values(by='similarity_score', ascending=False).head(3)

    st.subheader("Top Matching Jobs:")
    st.write(top_jobs[['job_description', 'similarity_score']])

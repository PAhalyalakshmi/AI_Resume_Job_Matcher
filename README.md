
# AI Resume & Job Matching System

## Project Overview

AI Resume & Job Matching System is an NLP-based web application
that analyzes a resume against a job description.

The application extracts resume text, calculates TF-IDF cosine
similarity, identifies predefined skills, and highlights potential
skill gaps.

## Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- NLP
- TF-IDF Vectorization
- Cosine Similarity
- PyPDF
- python-docx
- Plotly

## Features

- PDF and DOCX resume upload
- Resume text extraction
- Job description text input
- TF-IDF text similarity calculation
- Predefined skill extraction
- Matching skill identification
- Potential skill gap analysis
- Skill improvement recommendations
- Keyword frequency analysis
- Interactive score dashboard
- CSV report download

## Installation

Install Python 3.10 or later.

Create a virtual environment:

python -m venv .venv

Activate on Windows:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

## How It Works

1. Upload a resume in PDF or DOCX format.
2. Paste the job description.
3. Extract text from the resume.
4. Clean the text using regular expressions.
5. Convert resume and job description text into TF-IDF vectors.
6. Calculate cosine similarity.
7. Extract predefined skills.
8. Identify matched and missing skills.
9. Generate the matching dashboard.
10. Download the CSV report.

## Important Limitations

- Similarity is a text comparison estimate, not an official ATS score.
- Skill extraction uses a predefined dictionary.
- The system does not understand all context or synonyms.
- Recommendations are rule-based.
- Results should not be used as an automatic hiring decision.

## Future Improvements

- Semantic similarity using sentence embeddings
- LLM-powered resume recommendations
- PostgreSQL database integration
- Resume history tracking
- FastAPI REST API
- Cloud deployment
- PDF report generation

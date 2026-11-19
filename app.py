
import streamlit as st
import pandas as pd
import re
import io

from pypdf import PdfReader
from docx import Document

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import plotly.express as px


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Resume & Job Matcher",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume & Job Matching System")

st.markdown("""
An NLP-powered application that compares a resume with a job description,
calculates text similarity, identifies matching skills, and highlights
potential skill gaps.

**Note:** This tool provides a text and keyword similarity estimate.
It is not an official ATS score or a prediction of hiring success.
""")


# --------------------------------------------------
# SKILL DICTIONARY
# --------------------------------------------------

SKILL_DICTIONARY = [
    "python",
    "java",
    "c++",
    "sql",
    "mysql",
    "postgresql",
    "mongodb",
    "excel",
    "advanced excel",
    "pivot tables",
    "vlookup",
    "xlookup",
    "power bi",
    "tableau",
    "streamlit",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "natural language processing",
    "nlp",
    "data analytics",
    "data analysis",
    "data visualization",
    "data cleaning",
    "data validation",
    "data entry",
    "data management",
    "data quality",
    "mis reporting",
    "reporting",
    "dashboard",
    "dashboards",
    "business intelligence",
    "statistics",
    "statistics analysis",
    "forecasting",
    "anomaly detection",
    "automation",
    "etl",
    "power query",
    "powerpoint",
    "word",
    "communication",
    "problem solving",
    "analytical thinking",
    "attention to detail",
    "documentation",
    "customer service",
    "operations",
    "business analysis",
    "html",
    "css",
    "javascript",
    "react",
    "fastapi",
    "flask",
    "git",
    "github",
    "aws",
    "cloud computing",
    "rest api",
    "api",
    "docker",
    "linux",
    "agile",
    "project management"
]


# --------------------------------------------------
# RESUME TEXT EXTRACTION
# --------------------------------------------------

def extract_pdf_text(file):

    reader = PdfReader(file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def extract_docx_text(file):

    document = Document(file)

    text = ""

    for paragraph in document.paragraphs:

        text += paragraph.text + "\n"

    # Extract table content too
    for table in document.tables:

        for row in table.rows:

            for cell in row.cells:

                text += cell.text + " "

    return text


def extract_resume_text(uploaded_file):

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):

        return extract_pdf_text(uploaded_file)

    elif file_name.endswith(".docx"):

        return extract_docx_text(uploaded_file)

    else:

        raise ValueError(
            "Unsupported file format. Upload PDF or DOCX."
        )


# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z0-9+#.\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# --------------------------------------------------
# SKILL EXTRACTION
# --------------------------------------------------

def extract_skills(text):

    cleaned_text = clean_text(text)

    detected_skills = []

    for skill in SKILL_DICTIONARY:

        pattern = (
            r"(?<![a-zA-Z0-9])"
            + re.escape(skill.lower())
            + r"(?![a-zA-Z0-9])"
        )

        if re.search(
            pattern,
            cleaned_text
        ):

            detected_skills.append(skill)

    return sorted(
        set(detected_skills)
    )


# --------------------------------------------------
# TF-IDF SIMILARITY
# --------------------------------------------------

def calculate_similarity(
    resume_text,
    job_description
):

    resume_cleaned = clean_text(
        resume_text
    )

    job_cleaned = clean_text(
        job_description
    )

    if not resume_cleaned or not job_cleaned:

        return 0.0

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000
    )

    try:

        vectors = vectorizer.fit_transform(
            [
                resume_cleaned,
                job_cleaned
            ]
        )

        similarity = cosine_similarity(
            vectors[0:1],
            vectors[1:2]
        )[0][0]

        return float(similarity * 100)

    except ValueError:

        return 0.0


# --------------------------------------------------
# SKILL GAP ANALYSIS
# --------------------------------------------------

def analyze_skill_gap(
    resume_text,
    job_description
):

    resume_skills = extract_skills(
        resume_text
    )

    job_skills = extract_skills(
        job_description
    )

    matched_skills = sorted(
        set(resume_skills)
        .intersection(
            set(job_skills)
        )
    )

    missing_skills = sorted(
        set(job_skills)
        .difference(
            set(resume_skills)
        )
    )

    if len(job_skills) > 0:

        skill_match_percentage = (
            len(matched_skills)
            / len(job_skills)
        ) * 100

    else:

        skill_match_percentage = 0

    return (
        resume_skills,
        job_skills,
        matched_skills,
        missing_skills,
        skill_match_percentage
    )


# --------------------------------------------------
# KEYWORD FREQUENCY
# --------------------------------------------------

def get_keyword_frequency(text):

    cleaned_text = clean_text(text)

    frequency_data = []

    for skill in SKILL_DICTIONARY:

        pattern = (
            r"(?<![a-zA-Z0-9])"
            + re.escape(skill.lower())
            + r"(?![a-zA-Z0-9])"
        )

        count = len(
            re.findall(
                pattern,
                cleaned_text
            )
        )

        if count > 0:

            frequency_data.append({

                "Skill": skill,

                "Frequency": count

            })

    return pd.DataFrame(
        frequency_data
    ).sort_values(
        by="Frequency",
        ascending=False
    ) if frequency_data else pd.DataFrame(
        columns=["Skill", "Frequency"]
    )


# --------------------------------------------------
# GENERATE RECOMMENDATIONS
# --------------------------------------------------

def generate_recommendations(
    missing_skills
):

    recommendations = []

    if not missing_skills:

        recommendations.append(
            "The predefined skill dictionary found no missing skills."
        )

        recommendations.append(
            "Review the job description for responsibilities "
            "and other requirements not captured by the skill list."
        )

        return recommendations

    recommendations.append(
        "Review the missing skills listed below."
    )

    for skill in missing_skills:

        recommendations.append(
            f"Learn or practice {skill} if it is relevant "
            "to your career goals."
        )

    recommendations.append(
        "Only add skills to your resume after you can "
        "demonstrate them honestly."
    )

    return recommendations


# --------------------------------------------------
# DOWNLOAD CSV
# --------------------------------------------------

def create_download_report(
    similarity_score,
    skill_match_percentage,
    matched_skills,
    missing_skills
):

    report = pd.DataFrame({

        "Metric": [

            "Text Similarity Percentage",

            "Skill Match Percentage",

            "Matched Skills",

            "Missing Skills"

        ],

        "Result": [

            round(
                similarity_score,
                2
            ),

            round(
                skill_match_percentage,
                2
            ),

            ", ".join(
                matched_skills
            ),

            ", ".join(
                missing_skills
            )

        ]

    })

    output = io.StringIO()

    report.to_csv(
        output,
        index=False
    )

    return output.getvalue()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title(
    "About the Project"
)

st.sidebar.info("""
AI Resume & Job Matching System

Technologies:
- Python
- Streamlit
- NLP
- TF-IDF
- Cosine Similarity
- Scikit-learn
- Pandas

Purpose:
Resume and job description text analysis.
""")


# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.subheader("1. Upload Your Resume")

uploaded_resume = st.file_uploader(
    "Upload Resume",
    type=["pdf", "docx"]
)

st.subheader("2. Enter Job Description")

job_description = st.text_area(
    "Paste the complete job description here",
    height=300,
    placeholder="""
Example:

We are hiring an MIS Executive.

Required skills:
Excel, SQL, Power BI, Python,
data cleaning, reporting,
data validation, communication.
"""
)


# --------------------------------------------------
# ANALYSIS BUTTON
# --------------------------------------------------

if st.button(
    "Analyze Resume",
    type="primary",
    use_container_width=True
):

    if uploaded_resume is None:

        st.error(
            "Please upload a resume."
        )

    elif not job_description.strip():

        st.error(
            "Please enter a job description."
        )

    else:

        try:

            resume_text = extract_resume_text(
                uploaded_resume
            )

            if not resume_text.strip():

                st.error(
                    "No text could be extracted. "
                    "The PDF may be scanned or image-based."
                )

                st.stop()

            # ----------------------------------------
            # CALCULATE SIMILARITY
            # ----------------------------------------

            similarity_score = calculate_similarity(
                resume_text,
                job_description
            )

            (
                resume_skills,
                job_skills,
                matched_skills,
                missing_skills,
                skill_match_percentage
            ) = analyze_skill_gap(
                resume_text,
                job_description
            )

            # ----------------------------------------
            # DASHBOARD METRICS
            # ----------------------------------------

            st.divider()

            st.subheader(
                "3. Resume Matching Dashboard"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Text Similarity",
                f"{similarity_score:.2f}%"
            )

            col2.metric(
                "Skill Match",
                f"{skill_match_percentage:.2f}%"
            )

            col3.metric(
                "Matched Skills",
                len(matched_skills)
            )

            # ----------------------------------------
            # SCORE VISUALIZATION
            # ----------------------------------------

            score_df = pd.DataFrame({

                "Metric": [

                    "Text Similarity",

                    "Skill Match"

                ],

                "Percentage": [

                    similarity_score,

                    skill_match_percentage

                ]

            })

            fig = px.bar(

                score_df,

                x="Metric",

                y="Percentage",

                title="Resume Matching Scores",

                text="Percentage"

            )

            fig.update_layout(
                yaxis_range=[0, 100]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # ----------------------------------------
            # MATCHED SKILLS
            # ----------------------------------------

            st.subheader(
                "4. Matching Skills"
            )

            if matched_skills:

                st.success(
                    ", ".join(
                        matched_skills
                    )
                )

            else:

                st.warning(
                    "No predefined matching skills found."
                )

            # ----------------------------------------
            # MISSING SKILLS
            # ----------------------------------------

            st.subheader(
                "5. Potential Skill Gaps"
            )

            if missing_skills:

                st.warning(
                    ", ".join(
                        missing_skills
                    )
                )

                st.markdown(
                    "These are skills detected in the job description "
                    "but not detected in the resume."
                )

            else:

                st.success(
                    "No missing skills were detected "
                    "from the predefined skill list."
                )

            # ----------------------------------------
            # RECOMMENDATIONS
            # ----------------------------------------

            st.subheader(
                "6. Skill Improvement Recommendations"
            )

            recommendations = generate_recommendations(
                missing_skills
            )

            for recommendation in recommendations:

                st.write(
                    "• " + recommendation
                )

            # ----------------------------------------
            # RESUME SKILLS
            # ----------------------------------------

            st.subheader(
                "7. Skills Detected in Resume"
            )

            if resume_skills:

                st.write(
                    ", ".join(
                        resume_skills
                    )
                )

            else:

                st.info(
                    "No predefined skills detected in the resume."
                )

            # ----------------------------------------
            # JOB SKILLS
            # ----------------------------------------

            st.subheader(
                "8. Skills Detected in Job Description"
            )

            if job_skills:

                st.write(
                    ", ".join(
                        job_skills
                    )
                )

            else:

                st.info(
                    "No predefined skills detected in the job description."
                )

            # ----------------------------------------
            # KEYWORD FREQUENCY
            # ----------------------------------------

            st.subheader(
                "9. Job Description Keyword Analysis"
            )

            keyword_df = get_keyword_frequency(
                job_description
            )

            if not keyword_df.empty:

                st.dataframe(
                    keyword_df,
                    use_container_width=True
                )

                keyword_chart = px.bar(

                    keyword_df.head(15),

                    x="Skill",

                    y="Frequency",

                    title="Top Skill Keyword Frequencies"

                )

                st.plotly_chart(
                    keyword_chart,
                    use_container_width=True
                )

            # ----------------------------------------
            # DOWNLOAD REPORT
            # ----------------------------------------

            st.subheader(
                "10. Download Analysis Report"
            )

            report_csv = create_download_report(

                similarity_score,

                skill_match_percentage,

                matched_skills,

                missing_skills

            )

            st.download_button(

                label="Download Resume Matching Report CSV",

                data=report_csv,

                file_name="resume_job_matching_report.csv",

                mime="text/csv",

                use_container_width=True

            )

            # ----------------------------------------
            # EXTRACTED RESUME PREVIEW
            # ----------------------------------------

            with st.expander(
                "View Extracted Resume Text"
            ):

                st.text_area(

                    "Extracted Text",

                    resume_text,

                    height=400

                )

            st.success(
                "Resume analysis completed successfully."
            )

        except Exception as error:

            st.error(
                f"An error occurred: {error}"
            )

            st.info("""
            Check that:
            - The PDF or DOCX file is valid.
            - The file is not password protected.
            - The job description contains text.
            """)

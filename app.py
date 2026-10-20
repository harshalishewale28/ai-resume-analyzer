
import re
from io import BytesIO

import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

SKILLS = [
    "python", "java", "c++", "javascript", "html", "css",
    "sql", "mysql", "mongodb", "react", "node.js", "flask",
    "django", "streamlit", "machine learning", "deep learning",
    "artificial intelligence", "data analysis", "pandas",
    "numpy", "scikit-learn", "aws", "azure", "docker",
    "kubernetes", "linux", "git", "github", "jenkins",
    "ci/cd", "devops", "terraform", "networking",
    "rest api", "excel", "power bi", "tableau",
    "communication", "teamwork", "problem solving"
]


def extract_pdf_text(pdf_bytes):
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []

    for page in reader.pages:
        pages.append(page.extract_text() or "")

    return "\n".join(pages).strip()


def find_skills(text):
    text = text.lower()
    found = []

    for skill in SKILLS:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        if re.search(pattern, text):
            found.append(skill)

    return sorted(set(found))


def compare_texts(resume, job_description):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(
        [resume, job_description]
    )

    score = cosine_similarity(
        matrix[0:1], matrix[1:2]
    )[0][0]

    return round(score * 100, 1)


st.title("📄 AI Resume Analyzer")
st.write(
    "Analyze your resume, compare it with a job description, "
    "and identify skills you may need to develop."
)

st.info(
    "Privacy tip: Use a sample or redacted resume. "
    "This demo processes the uploaded PDF in the app session."
)

with st.sidebar:
    st.header("About this project")
    st.write("**Technology:** Python, Streamlit, NLP")
    st.write("**Features:** PDF extraction, skill matching, "
             "text similarity and recommendations.")
    st.caption(
        "The match score is an estimate, not an official ATS score "
        "or a prediction of hiring success."
    )

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload your resume")
    uploaded_file = st.file_uploader(
        "Choose a PDF resume",
        type=["pdf"]
    )

with col2:
    st.subheader("2. Add the job description")
    job_description = st.text_area(
        "Paste the full job description",
        height=240,
        placeholder=(
            "Example: We are looking for a Python developer "
            "with SQL, Git, AWS and problem-solving skills..."
        )
    )

if st.button("Analyze Resume", type="primary"):
    if uploaded_file is None:
        st.error("Please upload a PDF resume first.")
    elif not job_description.strip():
        st.error("Please paste a job description.")
    else:
        try:
            resume_text = extract_pdf_text(
                uploaded_file.getvalue()
            )

            if len(resume_text) < 30:
                st.error(
                    "Not enough readable text was found. "
                    "Try a text-based PDF instead of a scanned image."
                )
            else:
                resume_skills = find_skills(resume_text)
                job_skills = find_skills(job_description)

                matched = sorted(
                    set(resume_skills) & set(job_skills)
                )
                missing = sorted(
                    set(job_skills) - set(resume_skills)
                )

                score = compare_texts(
                    resume_text, job_description
                )

                coverage = (
                    len(matched) / len(job_skills) * 100
                    if job_skills else 0
                )

                st.divider()
                st.header("Analysis Results")

                a, b, c = st.columns(3)
                a.metric("Text similarity", f"{score}%")
                b.metric("Skills found", len(resume_skills))
                c.metric("Job skills matched", len(matched))

                st.progress(min(score / 100, 1.0))
                st.caption(
                    "Text similarity compares wording and term "
                    "importance; it does not measure your true "
                    "qualification or guarantee ATS performance."
                )

                st.subheader("Skills comparison")

                left, right = st.columns(2)

                with left:
                    st.markdown("**✅ Matching skills**")
                    if matched:
                        st.write(", ".join(matched))
                    else:
                        st.write("No matching listed skills found.")

                with right:
                    st.markdown("**📌 Skills not detected**")
                    if missing:
                        st.write(", ".join(missing))
                    else:
                        st.write(
                            "All skills in this basic skill list "
                            "were detected, or no listed skills "
                            "were found in the job description."
                        )

                st.subheader("Recommended next steps")

                if missing:
                    st.write(
                        "1. Review the missing skills and learn "
                        "the ones genuinely relevant to your goals."
                    )
                    st.write(
                        "2. Add relevant skills to your resume "
                        "only when you have those skills."
                    )
                    st.write(
                        "3. Build a project that demonstrates "
                        "the skills you actually know."
                    )
                else:
                    st.write(
                        "Review the job requirements carefully "
                        "and provide evidence of relevant skills "
                        "through projects and experience."
                    )

                st.subheader("Resume text preview")
                with st.expander("View extracted resume text"):
                    st.text(resume_text[:10000])

                report = (
                    "AI RESUME ANALYZER REPORT\n\n"
                    f"Text similarity: {score}%\n"
                    f"Job skill coverage: {coverage:.1f}%\n"
                    f"Resume skills: {', '.join(resume_skills)}\n"
                    f"Matched skills: {', '.join(matched)}\n"
                    f"Skills not detected: {', '.join(missing)}\n\n"
                    "Note: This is an estimate, not an official ATS score."
                )

                st.download_button(
                    "Download analysis report",
                    data=report,
                    file_name="resume_analysis.txt",
                    mime="text/plain"
                )

        except Exception as error:
            st.error(
                "The PDF could not be analyzed. "
                "Check that it is a valid, readable PDF."
            )
            st.caption(f"Technical detail: {error}")

st.divider()
st.caption("Built as a learning project using Python and NLP.")

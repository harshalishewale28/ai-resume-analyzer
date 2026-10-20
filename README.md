
# AI Resume Analyzer 📄

A Python web application that analyzes PDF resumes and compares
their content with a job description using NLP techniques.

## Features
- Upload a PDF resume
- Extract readable resume text
- Detect skills using a predefined skill list
- Compare resume and job description text
- Identify matching and missing listed skills
- Download a text analysis report

## Technologies
- Python
- Streamlit
- PyPDF
- Scikit-learn
- TF-IDF and cosine similarity

## Installation

1. Clone this repository.
2. Create a virtual environment:

   python -m venv .venv

3. Activate it on Windows:

   .venv\Scripts\activate

4. Install the dependencies:

   pip install -r requirements.txt

5. Run the application:

   streamlit run app.py

## Limitations
The similarity score is an estimate, not an official ATS score.
Skill detection uses a predefined list and may miss synonyms.
This initial version does not use a generative AI model.

## Future Improvements
- Integrate a real AI model
- Add more skills and synonym detection
- Generate downloadable PDF reports
- Improve resume recommendations

import pdfplumber
import docx
import re
# import nltk
# nltk.download('punkt',quiet=True)
# from nltk.tokenize import sent_tokenize
from sentence_transformers import util
from sklearn.metrics.pairwise import cosine_similarity

SKILL_KEYWORDS = [
    "python", "tensorflow", "pytorch", "scikit-learn", "sql", "aws", "azure", "docker", "kubernetes",
    "pandas", "numpy", "matplotlib", "seaborn", "nlp", "deep learning", "machine learning",
    "data analysis", "data visualization", "regression", "classification", "time series",
    "xgboost", "lightgbm", "feature engineering", "model deployment", "mlflow"
]

def split_sentences(text):
    text = re.sub(r'\n+', '\n', text.strip())

    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])|\n', text)

    return [s.strip() for s in parts if s.strip()]
def parse_resume(file):
    filename = file.name.lower()
    
    if filename.endswith('.pdf'):
        with pdfplumber.open(file) as pdf:
            text = "\n".join([page.extract_text() or '' for page in pdf.pages])
    
    elif filename.endswith('.docx'):
        doc = docx.Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
    
    elif filename.endswith('.txt'):
        text = file.read().decode("utf-8", errors="ignore")
    
    else:
        raise ValueError("Unsupported file format.")
    
    return text.strip()

def detect_missing_skills(resume_text, job_description, skill_list=SKILL_KEYWORDS):
    resume_text = resume_text.lower()
    job_description = job_description.lower()
    
    jd_list = [skill for skill in skill_list if skill in job_description]
    missing_skills = [skill for skill in jd_list if skill not in resume_text]
    
    return missing_skills

def semantic_coverage_score_sectionwise(resume_sections, job_description, model, threshold = 0.5):
    jd_sentences = split_sentences(job_description)

    matched_sentences = 0
    for sentence in jd_sentences:
        sentence_lower = sentence.lower()

        if ("skill" in sentence_lower or "tools" in sentence_lower or "proficient" in sentence_lower or "technologies" in sentence_lower):
            section_text = resume_sections['skills']
        elif ("experience" in sentence_lower or "develop" in sentence_lower or
              "implement" in sentence_lower or "work" in sentence_lower or
              "responsible" in sentence_lower or "design" in sentence_lower or
              "build" in sentence_lower or "deploy" in sentence_lower):
            section_text = resume_sections["experience"]

        else:
            section_text = resume_sections["other"]

        section_embedding = model.encode(section_text)
        sentence_embedding = model.encode(sentence)

        similarity = util.cos_sim(section_embedding, sentence_embedding)[0][0].item()
        if similarity >= threshold:
            matched_sentences +=1

    coverage_percent= round((matched_sentences/len(jd_sentences))*100,2)
    return coverage_percent, matched_sentences, len(jd_sentences)


def extract_resume_sections(resume_text):
    resume_text = resume_text.lower()
    sections = {
        "skills": "",
        "experience": "",
        "education": "",
        "other": ""
    }

    chunks = re.split(r'\n(?=[A-Z][a-z]+:|[A-Z\s]{3,})', resume_text, flags=re.IGNORECASE)

    for chunk in chunks:
        if 'skill' in chunk:
            sections['skills'] +=chunk
        elif "experience" in chunk or "work" in chunk:
            sections['experience'] += chunk
        elif "education" in chunk:
            sections['education'] += chunk
        else:
            sections['other'] += chunk
    
    return sections


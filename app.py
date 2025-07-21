import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from resume_parser import (
    parse_resume, 
    detect_missing_skills, 
    semantic_coverage_score_sectionwise, 
    extract_resume_sections,
    SKILL_KEYWORDS
)

def load_model():
    # return SentenceTransformer('all-MiniLM-L6-v2', device = 'cpu')
    # return SentenceTransformer('all-mpnet-base-v2', device = 'cpu')
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device = 'cpu')

model = load_model()    

st.set_page_config(page_title="Smart Resume-Job Matcher", layout = 'centered')
st.title(" Smart Resume-Job Matcher")

uploaded_file = st.file_uploader("Upload your resume", type = ['pdf', 'docx', 'txt'])

job_description = st.text_area('Paste the job description here')

resume_text = ''

if uploaded_file:
    try:
        resume_text = parse_resume(uploaded_file)
        st.subheader(" Extracted Resume Text ")
        st.text_area("Resume Text", resume_text, height = 250)
    except Exception as e:
        st.error(f"Error processing file: {e}")

if job_description:
    st.subheader("Job Description")
    st.text_area("Job Description Text", job_description, height=250)

if resume_text and job_description:
    with st.spinner("Calculating match score..."):
        resume_embedding = model.encode(resume_text)
        jd_embedding = model.encode(job_description)

        similarity = cosine_similarity([resume_embedding], [jd_embedding])[0][0]
        match_score = round(similarity*100,2)

        st.success("Resume and Job Description matched successfully!")
        st.metric(label = 'Match Score', value = f"{match_score}%")

    missing_skills = detect_missing_skills(resume_text, job_description, SKILL_KEYWORDS)

    if missing_skills:
        st.warning("Some important skills from the job description are missing in your resume:")
        st.write(", ".join(missing_skills))
    else:
        st.success("Your resume covers all the key skills mentioned in the job description")

    resume_sections = extract_resume_sections(resume_text)
    coverage_score, matched, total = semantic_coverage_score_sectionwise(resume_sections, job_description, model)
    st.subheader("JD Coverage Analysis.")
    st.write(f"Your resume semantically covers **{matched} out of {total}** job description statements.")
    st.metric(label = 'Semantic Coverage Score', value = f"{coverage_score}%")
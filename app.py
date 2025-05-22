import streamlit as st
import spacy
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from io import BytesIO
from PyPDF2 import PdfReader # For PDF parsing
from docx import Document # For DOCX parsing
from sentence_transformers import SentenceTransformer

# --- 1. Load NLP Models ---
@st.cache_resource # Cache the model loading for performance
def load_nlp_models():
    try:
        nlp = spacy.load("en_core_web_lg")
    except OSError:
        st.error("spaCy 'en_core_web_lg' model not found. Please run 'python -m spacy download en_core_web_lg' in your terminal.")
        st.stop() # Stop the app if model is not found
    
    # Using Sentence-Transformers for robust sentence embeddings
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    return nlp, sentence_model

nlp, sentence_model = load_nlp_models()

# --- 2. Text Extraction Functions ---
def extract_text_from_pdf(file_bytes):
    """Extracts text from a PDF file."""
    text = ""
    try:
        reader = PdfReader(BytesIO(file_bytes))
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
    return text

def extract_text_from_docx(file_bytes):
    """Extracts text from a DOCX file."""
    text = ""
    try:
        document = Document(BytesIO(file_bytes))
        for para in document.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {e}")
    return text

# --- 3. Text Preprocessing ---
def preprocess_text(text):
    """Cleans and preprocesses text for NLP."""
    text = text.lower() # Convert to lowercase
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
    # Consider adding more cleaning specific to your data, e.g., removing URLs, excessive punctuation
    return text

# --- 4. Information Extraction (using spaCy NER and basic regex) ---
def parse_resume(resume_text):
    """Parses resume text to extract structured information."""
    doc = nlp(resume_text)
    
    parsed_data = {
        "skills": [],
        "experience_keywords": [], # Keywords related to experience
        "education_keywords": [],
        "contact": {},
        "name": "",
        "summary": ""
    }

    # Basic Name extraction (spaCy PERSON entity)
    for ent in doc.ents:
        if ent.label_ == "PERSON" and len(ent.text.split()) <= 4 and not parsed_data["name"]: # Heuristic for name
            parsed_data["name"] = ent.text
            break # Assume first PERSON is the name

    # Very basic skill extraction (can be improved with a predefined skill list or custom NER)
    # This list should be comprehensive for relevant tech skills
    common_tech_skills = ["python", "java", "sql", "machine learning", "nlp", "aws", "docker", "kubernetes", 
                          "javascript", "react", "angular", "devops", "cloud", "api", "rest", "git", 
                          "agile", "scrum", "data analysis", "tableau", "excel", "c++", "c#", "go", 
                          "azure", "gcp", "tensorflow", "pytorch", "spark", "hadoop", "linux", "html", "css"]
    
    found_skills = []
    for skill in common_tech_skills:
        if re.search(r'\b' + re.escape(skill) + r'\b', resume_text, re.IGNORECASE):
            found_skills.append(skill)
    parsed_data["skills"] = list(set(found_skills))

    # Basic section parsing (very fragile, needs robust rules for real-world)
    # A better approach would be to use rule-based parsing on identified sections.
    # For now, just look for keywords within the whole text.
    
    # Experience keywords (look for verbs and nouns associated with work)
    experience_indicators = ["developed", "implemented", "managed", "led", "designed", "built", "created", "responsible for", "contributed to"]
    parsed_data["experience_keywords"] = [kw for kw in experience_indicators if kw in resume_text]

    # Education keywords
    education_indicators = ["bachelor", "master", "phd", "university", "college", "degree", "diploma"]
    parsed_data["education_keywords"] = [kw for kw in education_indicators if kw in resume_text]

    # Contact info (simple regex)
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', resume_text)
    if email_match:
        parsed_data["contact"]["email"] = email_match.group(0)
    
    phone_match = re.search(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', resume_text)
    if phone_match:
        parsed_data["contact"]["phone"] = phone_match.group(0)

    return parsed_data

def parse_job_description(job_description_text):
    """Parses job description text to extract structured information."""
    doc = nlp(job_description_text)
    
    parsed_data = {
        "required_skills": [],
        "responsibilities_keywords": [],
        "qualifications_keywords": [],
        "job_title": "",
        "company": "",
        "location": ""
    }

    # Basic entity extraction for job title, company, location
    for ent in doc.ents:
        if ent.label_ == "ORG" and not parsed_data["company"]:
            parsed_data["company"] = ent.text
        if ent.label_ == "GPE" and not parsed_data["location"]:
            parsed_data["location"] = ent.text
        # Job title is hard with generic NER; often extracted via regex patterns or is the first line
    
    # Attempt to extract job title from the first few lines
    lines = job_description_text.strip().split('\n')
    if lines:
        # A simple heuristic: assume the first non-empty line is the title
        parsed_data["job_title"] = lines[0].strip().split("Job Title: ")[-1]

    # Required skills (using the same list as resume parsing for consistency)
    common_tech_skills = ["python", "java", "sql", "machine learning", "nlp", "aws", "docker", "kubernetes", 
                          "javascript", "react", "angular", "devops", "cloud", "api", "rest", "git", 
                          "agile", "scrum", "data analysis", "tableau", "excel", "c++", "c#", "go", 
                          "azure", "gcp", "tensorflow", "pytorch", "spark", "hadoop", "linux", "html", "css",
                          "communication", "leadership", "problem solving", "teamwork", "analytical", "critical thinking"] # Add soft skills
    
    found_required_skills = []
    for skill in common_tech_skills:
        if re.search(r'\b' + re.escape(skill) + r'\b', job_description_text, re.IGNORECASE):
            found_required_skills.append(skill)
    parsed_data["required_skills"] = list(set(found_required_skills))

    # Responsibilities/Qualifications keywords (simple keyword matching)
    responsibility_indicators = ["design", "develop", "implement", "maintain", "manage", "lead", "collaborate", "deploy", "optimize", "build"]
    parsed_data["responsibilities_keywords"] = [kw for kw in responsibility_indicators if re.search(r'\b' + re.escape(kw) + r'\b', job_description_text, re.IGNORECASE)]

    qualification_indicators = ["experience", "strong", "proven", "proficient", "familiarity", "bachelor's", "master's", "degree"]
    parsed_data["qualifications_keywords"] = [kw for kw in qualification_indicators if re.search(r'\b' + re.escape(kw) + r'\b', job_description_text, re.IGNORECASE)]

    return parsed_data

# --- 5. Semantic Similarity and Scoring ---
def get_sentence_embeddings(text_list):
    """Generates sentence embeddings for a list of texts."""
    embeddings = sentence_model.encode(text_list)
    return embeddings

def calculate_compatibility_score(resume_text, job_description_text):
    """Calculates a compatibility score between a resume and job description."""
    preprocessed_resume = preprocess_text(resume_text)
    preprocessed_job_desc = preprocess_text(job_description_text)

    # Get embeddings for the entire documents for overall semantic similarity
    resume_embedding = get_sentence_embeddings([preprocessed_resume])[0]
    job_desc_embedding = get_sentence_embeddings([preprocessed_job_desc])[0]

    # Reshape for cosine_similarity (expects 2D arrays)
    resume_embedding = resume_embedding.reshape(1, -1)
    job_desc_embedding = job_desc_embedding.reshape(1, -1)

    overall_score = cosine_similarity(resume_embedding, job_desc_embedding)[0][0]

    # Incorporate skill matching for a weighted score
    parsed_resume = parse_resume(resume_text)
    parsed_job_desc = parse_job_description(job_description_text)

    resume_skills = set(s.lower() for s in parsed_resume["skills"])
    job_skills = set(s.lower() for s in parsed_job_desc["required_skills"])

    matched_skills = len(resume_skills.intersection(job_skills))
    total_job_skills = len(job_skills)
    
    skill_match_ratio = 0
    if total_job_skills > 0:
        skill_match_ratio = matched_skills / total_job_skills

    # Combine overall semantic score with skill match score (weights can be tuned)
    final_score = (overall_score * 0.7 + skill_match_ratio * 0.3) * 100
    
    return final_score # Convert to a percentage

# --- 6. Resume Improvement Recommendations ---
def get_resume_recommendations(resume_parsed, job_desc_parsed):
    """
    Generates basic recommendations to improve resume fit based on job description.
    """
    recommendations = []

    resume_skills = set(s.lower() for s in resume_parsed["skills"])
    job_skills = set(s.lower() for s in job_desc_parsed["required_skills"])

    missing_skills = job_skills - resume_skills
    if missing_skills:
        recommendations.append(f"**Missing Skills:** Consider adding or highlighting your experience with these skills: {', '.join(sorted(list(missing_skills)))}.")
    
    # Check for keyword overlap in experience/responsibilities
    resume_exp_kw = set(s.lower() for s in resume_parsed["experience_keywords"])
    job_resp_kw = set(s.lower() for s in job_desc_parsed["responsibilities_keywords"])
    
    if not resume_exp_kw.intersection(job_resp_kw):
        recommendations.append("Ensure your experience section uses action verbs and descriptions that align with the job's key responsibilities.")

    # General advice
    if not recommendations:
        recommendations.append("Your resume seems to be a good fit! Focus on tailoring your summary and bullet points to directly address the job's specific requirements.")
    else:
        recommendations.append("Review your resume's formatting and clarity to ensure key information is easily scannable.")
    
    return recommendations

# --- 7. Real-time Job Fetching (Simulated) ---
def simulate_job_fetching(query, location):
    """Simulates fetching job postings."""
    st.info(f"Simulating job fetching for '{query}' in '{location}'...")
    sample_jobs = [
        {"title": "Software Engineer", "company": "Innovate Solutions", "location": "Hyderabad", "description": "Develop scalable backend systems using Python, AWS, and REST APIs. Strong problem-solving skills required. Experience with Docker and Kubernetes a plus."},
        {"title": "Data Scientist (NLP)", "company": "CogniTech Labs", "location": "Bangalore", "description": "Build and deploy NLP models for text classification and sentiment analysis. Expertise in Python, machine learning, and deep learning frameworks (TensorFlow/PyTorch). Excellent communication skills."},
        {"title": "DevOps Engineer", "company": "CloudBurst Inc.", "location": "Remote", "description": "Manage CI/CD pipelines, automate infrastructure on Azure/GCP, and ensure system reliability. Proficient in Linux, Docker, and Kubernetes. Agile environment experience preferred."},
        {"title": "Frontend Developer", "company": "Pixel Perfect Studio", "location": "Hyderabad", "description": "Design and implement responsive web UIs using React, JavaScript, HTML, and CSS. Collaborate with UX/UI designers. Experience with Git and agile methodologies."},
    ]
    
    # Filter simulated jobs by query and location (very basic)
    filtered_jobs = []
    query_lower = query.lower()
    location_lower = location.lower()
    for job in sample_jobs:
        if (query_lower in job['title'].lower() or query_lower in job['description'].lower()) and \
           (location_lower in job['location'].lower() or location_lower == "any" or job['location'].lower() == "remote"):
            filtered_jobs.append(job)
    return filtered_jobs

# --- Streamlit UI ---
st.set_page_config(page_title="AI-Powered HR ATS", layout="wide")

st.title("🤖 AI-Powered HR ATS: Resume & Job Matcher")

st.markdown("""
Welcome to the AI-Powered HR ATS! This tool helps you **match resumes with job descriptions**,
calculate **compatibility scores**, and provides **recommendations** for improving resume fit.
You can also explore **simulated job postings**.
""")

st.sidebar.header("Upload Documents")
uploaded_resume = st.sidebar.file_uploader("Upload your Resume (PDF or DOCX)", type=["pdf", "docx"])
job_description_input = st.sidebar.text_area("Paste Job Description Here", height=300, 
                                            value="""Job Title: Software Engineer (Python/AWS)
Company: Innovate Corp
Location: Remote
About Us:
Innovate Corp is a fast-growing tech company building the next generation of AI products.

Responsibilities:
- Design, develop, and deploy highly scalable backend services using Python.
- Work extensively with AWS services including Lambda, S3, DynamoDB, and EC2.
- Implement robust data processing pipelines.
- Mentor junior engineers and contribute to architectural decisions.

Qualifications:
- 5+ years of professional experience in backend development.
- Strong expertise in Python and modern frameworks (e.g., FastAPI, Django).
- Proven experience with AWS cloud platform.
- Familiarity with containerization (Docker, Kubernetes).
- Excellent communication and problem-solving skills.
- Bachelor's or Master's degree in Computer Science or related field.
""")

st.markdown("---")

st.header("Resume & Job Description Analysis")

resume_text = ""
if uploaded_resume:
    if uploaded_resume.type == "application/pdf":
        resume_text = extract_text_from_pdf(uploaded_resume.read())
    elif uploaded_resume.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume_text = extract_text_from_docx(uploaded_resume.read())
    
    if resume_text:
        st.subheader("Parsed Resume Details:")
        parsed_resume_data = parse_resume(resume_text)
        st.json(parsed_resume_data)
    else:
        st.warning("Could not extract text from the uploaded resume.")

job_desc_text = job_description_input
if job_desc_text:
    st.subheader("Parsed Job Description Details:")
    parsed_job_desc_data = parse_job_description(job_desc_text)
    st.json(parsed_job_desc_data)
else:
    st.warning("Please paste a job description.")

st.markdown("---")

if resume_text and job_desc_text:
    st.header("Compatibility Score & Recommendations")
    
    compatibility_score = calculate_compatibility_score(resume_text, job_desc_text)
    st.markdown(f"### Overall Compatibility Score: **{compatibility_score:.2f}%**")
    
    st.subheader("Resume Improvement Recommendations:")
    recommendations = get_resume_recommendations(parsed_resume_data, parsed_job_desc_data)
    for i, rec in enumerate(recommendations):
        st.write(f"• {rec}")
else:
    st.info("Upload a resume and paste a job description to see the compatibility score and recommendations.")

st.markdown("---")

st.header("Simulated Job Fetching")
st.markdown("**(Note: This section simulates job fetching due to API limitations with real job boards.)**")

col1, col2 = st.columns(2)
with col1:
    job_query = st.text_input("Job Title / Keywords", "Software Engineer")
with col2:
    job_location = st.text_input("Location (e.g., Hyderabad, Bangalore, Remote, Any)", "Any")

if st.button("Search Simulated Jobs"):
    simulated_jobs = simulate_job_fetching(job_query, job_location)
    if simulated_jobs:
        st.subheader(f"Found {len(simulated_jobs)} Simulated Jobs:")
        for job in simulated_jobs:
            st.markdown(f"**{job['title']}** at **{job['company']}** ({job['location']})")
            with st.expander("Job Description"):
                st.write(job['description'])
            st.markdown("---")
    else:
        st.info("No simulated jobs found for your criteria. Try different keywords.")

st.markdown("""
---
*Disclaimer: This is a simplified prototype. Real-world ATS and HR software
require robust parsing, advanced NLP models (often fine-tuned LLMs), and
secure, legal integrations with external services.*
""")

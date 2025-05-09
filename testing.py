# app.py (Complete Version)
import os
import streamlit as st
import pdfplumber
import spacy
import imaplib
import email
import hashlib
import re
from email.header import decode_header
from datetime import datetime
import plotly.express as px
from dotenv import load_dotenv
from spacy.matcher import PhraseMatcher

# Load environment variables
load_dotenv()

# Configuration
CV_REPO_DIR = "cv_repository"
os.makedirs(CV_REPO_DIR, exist_ok=True)

# Load models
@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_trf")
    return nlp

nlp = load_models()

# --------------------------
# Email Processing
# --------------------------
def fetch_cvs_from_email():
    mail = imaplib.IMAP4_SSL(os.getenv("IMAP_SERVER", "imap.gmail.com"))
    mail.login(os.getenv("EMAIL"), os.getenv("EMAIL_PASSWORD"))
    mail.select("inbox")
    
    status, messages = mail.search(None, '(UNSEEN)')
    email_ids = messages[0].split()
    
    for e_id in email_ids:
        _, msg_data = mail.fetch(e_id, "(RFC822)")
        raw_email = msg_data[0][1]
        msg = email.message_from_bytes(raw_email)
        
        for part in msg.walk():
            if part.get_content_maintype() == 'multipart':
                continue
            if part.get('Content-Disposition') is None:
                continue

            filename = part.get_filename()
            if filename:
                filename = decode_header(filename)[0][0]
                if isinstance(filename, bytes):
                    filename = filename.decode()
                filepath = os.path.join(CV_REPO_DIR, filename)
                
                if not os.path.exists(filepath):
                    with open(filepath, 'wb') as f:
                        f.write(part.get_payload(decode=True))

# --------------------------
# PDF Processing
# --------------------------
@st.cache_data(ttl=3600, show_spinner="Processing PDF...")
def process_pdf(filepath):
    with pdfplumber.open(filepath) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages])

# --------------------------
# JD Processing
# --------------------------
def process_jd(uploaded_file):
    jd_text = process_pdf(uploaded_file)
    doc = nlp(jd_text)
    
    requirements = {
        "skills": list(set([chunk.text for chunk in doc.noun_chunks 
                          if "skill" in chunk.root.text.lower()])),
        "experience": max([int(num) for num in re.findall(r"\d+", jd_text)[:2]] or [0]),
        "certifications": list(set(re.findall(r"\b[A-Z]{3,}\b", jd_text))),
        "education": list(set([sent.text for sent in doc.sents 
                              if any(kw in sent.text.lower() 
                                     for kw in ["degree", "education"])]))
    }
    return requirements

# --------------------------
# CV Analysis
# --------------------------
def analyze_cv(cv_path, requirements):
    cv_text = process_pdf(cv_path)
    doc = nlp(cv_text)
    
    # Experience extraction
    experience = max([int(num) for num in re.findall(r"\d+", cv_text)[:2]] or [0])
    
    # Skill matching
    matcher = PhraseMatcher(nlp.vocab)
    patterns = [nlp.make_doc(skill) for skill in requirements["skills"]]
    matcher.add("SKILLS", patterns)
    matches = matcher(doc)
    matched_skills = list(set([doc[start:end].text for _, start, end in matches]))
    
    # Score calculation
    skill_score = len(matched_skills)/len(requirements["skills"]) if requirements["skills"] else 0
    exp_score = min(experience/requirements["experience"], 1) if requirements["experience"] > 0 else 0
    total_score = (0.6 * skill_score + 0.4 * exp_score) * 100
    
    return {
        "name": os.path.basename(cv_path),
        "score": total_score,
        "experience": experience,
        "skills_matched": matched_skills,
        "certifications": list(set(re.findall(r"\b[A-Z]{3,}\b", cv_text))),
        "education": [sent.text for sent in doc.sents 
                      if any(kw in sent.text.lower() 
                             for kw in ["degree", "education"])]
    }

# --------------------------
# Visualization
# --------------------------
def show_score_breakdown(results):
    df = {
        "Category": ["Skills", "Experience", "Certifications", "Education"],
        "Score": [
            sum(r['score'] * 0.6 for r in results)/len(results),
            sum(r['score'] * 0.4 for r in results)/len(results),
            sum(len(r['certifications']) for r in results)/len(results),
            sum(1 for r in results if r['education'])/len(results)*100
        ]
    }
    fig = px.bar(df, x='Category', y='Score', title="Score Breakdown Across All CVs")
    st.plotly_chart(fig)

# --------------------------
# Streamlit UI
# --------------------------
def main():
    st.set_page_config(page_title="CV Analyzer Pro", layout="wide")
    
    st.title("📊 Automated CV Analysis Suite")
    st.write("""
    **Complete workflow:**  
    1. Fetch CVs from email → 2. Upload JD → 3. Analyze → 4. Visualize
    """)
    
    # Sidebar Controls
    with st.sidebar:
        st.header("Controls")
        if st.button("🔄 Fetch New CVs"):
            with st.spinner("Syncing with email..."):
                fetch_cvs_from_email()
            st.success(f"{len(os.listdir(CV_REPO_DIR))} CVs in repository")
        
        jd_file = st.file_uploader("Upload Job Description", type="pdf")
    
    # Main Interface
    if jd_file:
        jd_requirements = process_jd(jd_file)
        
        with st.expander("Job Requirements Summary", expanded=True):
            cols = st.columns(4)
            cols[0].metric("Required Skills", len(jd_requirements["skills"]))
            cols[1].metric("Experience Needed", f"{jd_requirements['experience']}+ years")
            cols[2].metric("Certifications", len(jd_requirements["certifications"]))
            cols[3].metric("Education", "Required" if jd_requirements["education"] else "Optional")
        
        cv_files = [os.path.join(CV_REPO_DIR, f) for f in os.listdir(CV_REPO_DIR) if f.endswith(".pdf")]
        
        if cv_files:
            results = []
            progress_bar = st.progress(0)
            for i, cv_path in enumerate(cv_files):
                progress_bar.progress((i+1)/len(cv_files))
                results.append(analyze_cv(cv_path, jd_requirements))
            
            # Visualization Section
            st.header("Analysis Dashboard")
            show_score_breakdown(results)
            
            # Top Candidates
            st.subheader("🌟 Top 5 Candidates")
            top_candidates = sorted(results, key=lambda x: x['score'], reverse=True)[:5]
            for candidate in top_candidates:
                with st.container():
                    cols = st.columns([1,3,2])
                    cols[0].write(f"**{candidate['name']}**")
                    cols[1].metric("Score", f"{candidate['score']:.1f}%")
                    cols[2].progress(candidate['score']/100)
                    
                    with st.expander("Detailed Analysis"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Skills Matched:**", ", ".join(candidate['skills_matched']))
                            st.write("**Certifications:**", ", ".join(candidate['certifications']))
                        with col2:
                            st.write("**Experience:**", f"{candidate['experience']} years")
                            st.write("**Education:**", candidate['education'][0][:100] + "..." if candidate['education'] else "N/A")
            
            # Full Data Table
            st.subheader("📋 Complete Results")
            st.dataframe(
                sorted(results, key=lambda x: x['score'], reverse=True),
                column_config={
                    "score": st.column_config.ProgressColumn(
                        "Match %",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100
                    )
                },
                use_container_width=True
            )
        else:
            st.warning("No CVs found in repository! Fetch CVs first.")

if __name__ == "__main__":
    main()
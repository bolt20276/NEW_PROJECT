# app.py (Fixed Version)
import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import os
import streamlit as st
import pdfplumber
import spacy
import imaplib
import email
import re
import logging
from datetime import datetime, timedelta
import plotly.express as px
from email.header import decode_header
from dotenv import load_dotenv
from spacy.matcher import PhraseMatcher

# SET PAGE CONFIG FIRST
st.set_page_config(
    page_title="CV Analyzer Pro",
    layout="wide",
    page_icon="📄"
)

# Load environment variables
load_dotenv()

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
CV_REPO_DIR = "cv_repository"
os.makedirs(CV_REPO_DIR, exist_ok=True)

# Load models
@st.cache_resource
def load_models():
    return spacy.load("en_core_web_trf")

nlp = load_models()

# --------------------------
# Email Processing (Fixed)
# --------------------------
def fetch_cvs_from_email(start_date, end_date):
    """Enhanced version with proper date handling"""
    try:
        mail = imaplib.IMAP4_SSL(os.getenv("IMAP_SERVER", "imap.gmail.com"))
        mail.login(os.getenv("EMAIL"), os.getenv("EMAIL_PASSWORD"))
        
        # Check all folders
        mail.select("inbox")
        logger.info("Searching INBOX")
        
        # Format dates correctly
        imap_date_format = "%d-%b-%Y"
        search_query = (
            f'(SINCE "{start_date.strftime(imap_date_format)}" '
            f'BEFORE "{(end_date + timedelta(days=1)).strftime(imap_date_format)}")'
            ' (HASATTACH)'
        )
        
        status, messages = mail.search(None, search_query)
        email_ids = messages[0].split()
        fetched_cvs = 0
        
        for e_id in email_ids:
            _, msg_data = mail.fetch(e_id, "(RFC822)")
            msg = email.message_from_bytes(msg_data[0][1])
            
            for part in msg.walk():
                if part.get_content_maintype() == 'multipart':
                    continue
                    
                if part.get('Content-Disposition') is None:
                    continue
                
                filename = part.get_filename()
                if filename:
                    # Decode filename properly
                    filename = decode_header(filename)[0][0]
                    if isinstance(filename, bytes):
                        filename = filename.decode(errors='ignore')
                    
                    # Save all PDF attachments
                    if filename.lower().endswith('.pdf'):
                        filepath = os.path.join(CV_REPO_DIR, filename)
                        if not os.path.exists(filepath):
                            with open(filepath, 'wb') as f:
                                f.write(part.get_payload(decode=True))
                            fetched_cvs += 1
        
        return fetched_cvs
    
    except Exception as e:
        logger.error(f"Fetch error: {str(e)}")
        return 0

# --------------------------
# Rest of Processing Functions (Unchanged)
# --------------------------
@st.cache_data(ttl=3600)
def process_pdf(filepath):
    with pdfplumber.open(filepath) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages])

def process_jd(uploaded_file):
    jd_text = process_pdf(uploaded_file)
    doc = nlp(jd_text)
    
    return {
        "skills": list({chunk.text for chunk in doc.noun_chunks if "skill" in chunk.root.text.lower()}),
        "experience": max([int(num) for num in re.findall(r"\d+", jd_text)[:2]] or [0]),
        "certifications": list(set(re.findall(r"\b[A-Z]{3,}\b", jd_text))),
        "education": list({sent.text for sent in doc.sents if any(kw in sent.text.lower() for kw in ["degree", "education"])})
    }

def analyze_cv(cv_path, requirements):
    cv_text = process_pdf(cv_path)
    doc = nlp(cv_text)
    
    matcher = PhraseMatcher(nlp.vocab)
    patterns = [nlp.make_doc(skill) for skill in requirements["skills"]]
    matcher.add("SKILLS", patterns)
    
    return {
        "name": os.path.basename(cv_path),
        "score": (0.6 * len(matcher(doc))/len(requirements["skills"]) + 0.4 * min(
            max([int(num) for num in re.findall(r"\d+", cv_text)[:2]] or [0])/requirements["experience"], 1
        )) * 100 if requirements["experience"] > 0 else 0,
        "experience": max([int(num) for num in re.findall(r"\d+", cv_text)[:2]] or [0]),
        "skills_matched": list(set([doc[start:end].text for _, start, end in matcher(doc)])),
        "certifications": list(set(re.findall(r"\b[A-Z]{3,}\b", cv_text))),
        "education": [sent.text for sent in doc.sents if any(kw in sent.text.lower() for kw in ["degree", "education"])]
    }

# --------------------------
# Streamlit UI (Fixed)
# --------------------------
def main():
    st.title("📊 Automated CV Analysis Suite")
    
    with st.sidebar:
        st.header("Controls")
        start_date = st.date_input("Start Date", datetime.today() - timedelta(days=7))
        end_date = st.date_input("End Date", datetime.today())
        
        if st.button("📩 Fetch CVs from Email"):
            if start_date > end_date:
                st.error("End date must be after start date!")
            else:
                with st.spinner("Fetching CVs..."):
                    count = fetch_cvs_from_email(start_date, end_date)
                    st.success(f"Fetched {count} new CVs!")
        
        jd_file = st.file_uploader("Upload Job Description", type="pdf")
    
    if jd_file:
        requirements = process_jd(jd_file)
        cv_files = [os.path.join(CV_REPO_DIR, f) for f in os.listdir(CV_REPO_DIR) if f.endswith(".pdf")]
        
        if cv_files:
            results = []
            progress_bar = st.progress(0)
            for i, cv_path in enumerate(cv_files):
                progress_bar.progress((i+1)/len(cv_files))
                results.append(analyze_cv(cv_path, requirements))
            
            # Display results
            st.header("Analysis Results")
            st.dataframe(
                sorted(results, key=lambda x: x["score"], reverse=True),
                column_config={
                    "score": st.column_config.ProgressColumn(
                        "Match %",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100
                    )
                }
            )
        else:
            st.warning("No CVs found! Fetch CVs first.")

if __name__ == "__main__":
    main()
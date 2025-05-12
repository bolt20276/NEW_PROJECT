# # # main.py
# # import streamlit as st
# # import pandas as pd
# # from document_processor import DocumentProcessor
# # from ai_processor import AIProcessor
# # from semantic_analyzer import SemanticAnalyzer
# # from utils import extract_text_from_pdf, extract_text_from_docx, create_download_link
# # import os
# # from config import Config
# # from models import CandidateProfile
# # import datetime

# # def main_page():
# #     st.title("📄 AI-Powered CV Analyzer")
    
# #     # Date Range Selection
# #     col1, col2 = st.columns(2)
# #     with col1:
# #         start_date = st.date_input("Start Date", 
# #                                   value=datetime.date.today() - datetime.timedelta(days=30))
# #     with col2:
# #         end_date = st.date_input("End Date", 
# #                                 value=datetime.date.today())
    
# #     # Keyword Search
# #     keywords = st.text_input("Search Keywords (comma-separated)", 
# #                            help="Example: resume, CV, application")
    
# #     # Email and processing inputs...
# #     st.subheader("Email Credentials")
# #     email_user = st.text_input("Email Address", help="Enter your email address")
# #     email_pass = st.text_input("Email Password", type="password", help="Enter your email password")
    
# #     if st.button("🚀 Analyze Resumes"):
# #         if not all([email_user, email_pass, jd_text.strip()]):
# #             st.error("❌ Please fill all required fields")
# #             return
            
# #         with st.spinner("🔍 Scanning emails and analyzing resumes..."):
# #             try:
# #                 # Convert dates to IMAP format
# #                 start_imap = start_date.strftime("%d-%b-%Y")
# #                 end_imap = end_date.strftime("%d-%b-%Y")
                
# #                 # Process keywords
# #                 keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
                
# #                 # Fetch emails with date and keyword filters
# #                 email_handler = EmailHandler(email_user, email_pass)
# #                 attachments = email_handler.fetch_attachments(
# #                     start_date=start_imap,
# #                     end_date=end_imap,
# #                     keywords=keyword_list
# #                 )
# #             except Exception as e:
# #                 st.error(f"⚠️ Error fetching emails: {str(e)}")
# #                 return
    
# #     # Job Description Input
# #     st.subheader("Job Requirements")
# #     jd_text = st.text_area("Paste Job Description", height=200)
# #     uploaded_jd = st.file_uploader("Upload Job Description (PDF/DOCX)", type=["pdf", "docx"])
    
# #     # Process JD upload
# #     if uploaded_jd:
# #         file_content = uploaded_jd.read()
# #         if uploaded_jd.name.endswith(".pdf"):
# #             jd_text = extract_text_from_pdf(file_content)
# #         else:
# #             jd_text = extract_text_from_docx(file_content)
    
# #     # Analysis Controls
# #     if st.button("🚀 Analyze Resumes"):
# #         if not all([email_user, email_pass, jd_text.strip()]):
# #             st.error("❌ Please fill all required fields")
# #             return
            
# #         with st.spinner("🔍 Scanning emails and analyzing resumes..."):
# #             try:
# #                 # Initialize processors
# #                 doc_processor = DocumentProcessor()
# #                 ai_processor = AIProcessor()
# #                 semantic_analyzer = SemanticAnalyzer(jd_text)
                
# #                 # Fetch email attachments
# #                 email_handler = EmailHandler(email_user, email_pass)
# #                 attachments = email_handler.fetch_attachments()
                
# #                 candidates = []
# #                 for attachment in attachments:
# #                     try:
# #                         # Process document
# #                         docs = doc_processor.process_file(attachment['content'], 
# #                                                         os.path.splitext(attachment['filename'])[1])
                        
# #                         # Extract candidate profile
# #                         extraction_chain = ai_processor.create_extraction_chain()
# #                         profile = extraction_chain.run({"resume_text": docs[0].page_content})
                        
# #                         # Calculate scores
# #                         scoring_chain = ai_processor.create_scoring_chain(jd_text)
# #                         score_result = scoring_chain.run({"candidate_profile": profile.json()})
                        
# #                         # Semantic analysis
# #                         similarity_score = semantic_analyzer.calculate_similarity(docs[0].page_content)
# #                         final_score = 0.7*score_result.score + 0.3*similarity_score*100
                        
# #                         # Generate download link
# #                         download_link = create_download_link(
# #                             attachment['content'],
# #                             attachment['filename']
# #                         )
                        
# #                         candidates.append({
# #                             **profile.dict(),
# #                             **score_result.dict(),
# #                             "similarity_score": similarity_score,
# #                             "final_score": final_score,
# #                             "download": download_link
# #                         })
                        
# #                     except Exception as e:
# #                         st.error(f"⚠️ Error processing {attachment['filename']}: {str(e)}")
# #                         continue
                
# #                 if candidates:
# #                     st.session_state.candidates = candidates
# #                     st.session_state.show_results = True
# #                     st.rerun()
# #                 else:
# #                     st.warning("🤷 No valid resumes found in attachments")
                    
# #             except EmailFetchError as e:
# #                 st.error(f"📧 Email error: {str(e)}")
# #             except Exception as e:
# #                 st.error(f"⚡ Unexpected error: {str(e)}")

# # # main.py
# # # main.py
# # import streamlit as st
# # import pandas as pd
# # import os
# # import datetime
# # from document_processor import DocumentProcessor
# # from ai_processor import AIProcessor
# # from semantic_analyzer import SemanticAnalyzer
# # from email_handler import EmailHandler, EmailFetchError
# # from utils import extract_text_from_pdf, extract_text_from_docx, create_download_link
# # from config import Config
# # from models import CandidateProfile

# # def main_page():
# #     st.title("📄 AI-Powered CV Analyzer")
    
# #     # Initialize variables
# #     jd_text = ""
    
# #     # Job Description Section - MOVE TO TOP
# #     st.subheader("Job Requirements")
# #     jd_text = st.text_area("Paste Job Description", height=200, value="")
# #     uploaded_jd = st.file_uploader("Upload Job Description (PDF/DOCX)", type=["pdf", "docx"])
    
# #     # Process JD upload
# #     if uploaded_jd:
# #         file_content = uploaded_jd.read()
# #         if uploaded_jd.name.endswith(".pdf"):
# #             jd_text = extract_text_from_pdf(file_content)
# #         else:
# #             jd_text = extract_text_from_docx(file_content)

# #     # Email Configuration
# #     st.subheader("Email Credentials")
# #     col1, col2 = st.columns(2)
# #     with col1:
# #         email_user = st.text_input("Email Address", help="Enter your email address")
# #     with col2:
# #         email_pass = st.text_input("Password", type="password", help="Enter your email password")

# #     # Date Range Selection
# #     st.subheader("Search Filters")
# #     col1, col2 = st.columns(2)
# #     with col1:
# #         start_date = st.date_input("Start Date", 
# #                                   value=datetime.date.today() - datetime.timedelta(days=30))
# #     with col2:
# #         end_date = st.date_input("End Date", 
# #                                 value=datetime.date.today())
    
# #     # Keyword Search
# #     keywords = st.text_input("Search Keywords (comma-separated)", 
# #                            help="Example: resume, CV, application")

# #     # SINGLE Analyze Button with proper validation
# #     if st.button("🚀 Analyze Resumes", key="analyze_button"):
# #         if not all([email_user, email_pass]) or not jd_text.strip():
# #             st.error("❌ Please fill all required fields")
# #             return
            
# #         with st.spinner("🔍 Scanning emails and analyzing resumes..."):
# #             try:
# #                 # Convert dates to IMAP format
# #                 start_imap = start_date.strftime("%d-%b-%Y")
# #                 end_imap = end_date.strftime("%d-%b-%Y")
                
# #                 # Process keywords
# #                 keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
                
# #                 # Initialize processors
# #                 doc_processor = DocumentProcessor()
# #                 ai_processor = AIProcessor()
# #                 semantic_analyzer = SemanticAnalyzer(jd_text)
                
# #                 # Initialize email handler and fetch attachments
# #                 email_handler = EmailHandler(email_user, email_pass)
# #                 attachments = email_handler.fetch_attachments(
# #                     start_date=start_imap,
# #                     end_date=end_imap,
# #                     keywords=keyword_list,
# #                     max_emails=50
# #                 )
                
# #                 candidates = []
                
# #                 for attachment in attachments:
# #                     try:
# #                         # File validation
# #                         ext = os.path.splitext(attachment['filename'])[1].lower()
# #                         if ext not in ['.pdf', '.docx']:
# #                             st.warning(f"Skipped {attachment['filename']} - Invalid file type")
# #                             continue
                            
# #                         # Process document
# #                         docs = doc_processor.process_file(attachment['content'], ext)
# #                         if len(docs[0].page_content) < 300:
# #                             st.warning(f"Skipped {attachment['filename']} - Insufficient text")
# #                             continue
                            
# #                         # AI processing
# #                         extraction_chain = ai_processor.create_extraction_chain()
# #                         profile = extraction_chain.run({"resume_text": docs[0].page_content})
                        
# #                         # Scoring
# #                         scoring_chain = ai_processor.create_scoring_chain(jd_text)
# #                         score_result = scoring_chain.run({"candidate_profile": profile.json()})
                        
# #                         # Semantic analysis
# #                         similarity_score = semantic_analyzer.calculate_similarity(docs[0].page_content)
# #                         final_score = 0.7*score_result.score + 0.3*similarity_score*100
                        
# #                         # Store results
# #                         candidates.append({
# #                             **profile.dict(),
# #                             **score_result.dict(),
# #                             "similarity_score": similarity_score,
# #                             "final_score": final_score,
# #                             "download": create_download_link(
# #                                 attachment['content'],
# #                                 attachment['filename']
# #                             )
# #                         })
                        
# #                     except Exception as e:
# #                         st.error(f"⚠️ Error processing {attachment['filename']}: {str(e)}")
# #                         continue
                
# #                 if candidates:
# #                     st.session_state.candidates = candidates
# #                     st.session_state.show_results = True
# #                     st.rerun()
# #                 else:
# #                     st.warning("🤷 No valid resumes found in attachments")
                    
# #             except EmailFetchError as e:
# #                 st.error(f"📧 Email error: {str(e)}")
# #             except Exception as e:
# #                 st.error(f"⚡ Unexpected error: {str(e)}")

# # # Keep the results_page() and __main__ block unchanged from your original code


# # main.py
# import streamlit as st
# import pandas as pd
# from document_processor import DocumentProcessor
# from ai_processor import AIProcessor
# from semantic_analyzer import SemanticAnalyzer
# from utils import extract_text_from_pdf, extract_text_from_docx, create_download_link
# from email_handler import EmailHandler, EmailFetchError  
# import os
# from config import Config
# from models import CandidateProfile
# import datetime

# def main_page():
#     st.title("📄 AI-Powered CV Analyzer")
#     if st.checkbox("Using Corporate Network"):
#         os.environ["USE_CORPORATE_NETWORK"] = "True"
        
#     # Initialize variables first
#     jd_text = ""
    
#     # ========== Job Description Section ========== 
#     st.subheader("Job Requirements")
    
#     # Text input with unique key
#     jd_text = st.text_area(
#         "Paste Job Description", 
#         height=200, 
#         value="",
#         key="main_job_description_text"
#     )
    
#     # File uploader with unique key
#     uploaded_jd = st.file_uploader(
#         "Upload Job Description (PDF/DOCX/TXT)",
#         type=["pdf", "docx", "txt"],
#         key="job_description_upload_widget"
#     )
    
#     # Process JD upload
#     if uploaded_jd:
#         file_content = uploaded_jd.read()
#         if uploaded_jd.name.endswith(".pdf"):
#             jd_text = extract_text_from_pdf(file_content)
#         elif uploaded_jd.name.endswith(".docx"):
#             jd_text = extract_text_from_docx(file_content)
#         elif uploaded_jd.name.endswith(".txt"):
#             jd_text = file_content.decode("utf-8")

#     # ========== Connection Troubleshooting ==========
#     with st.expander("🔧 Connection Troubleshooting Guide"):
#         st.markdown("""
#         **Common Fixes:**
#         1. Enable IMAP in your email account settings
#         2. For Gmail: [Enable IMAP](https://mail.google.com/mail/#settings/fwdandpop)
#         3. Use App Password if 2FA enabled
#         4. Check firewall/antivirus isn't blocking port 993
#         5. Try different network (e.g., switch from WiFi to mobile hotspot)
#         """)

#     # ========== Email Credentials ==========
#     st.subheader("Email Credentials")
#     col1, col2 = st.columns(2)
#     with col1:
#         email_user = st.text_input("Email Address", key="email_input")
#     with col2:
#         email_pass = st.text_input("Password", type="password", key="password_input")

#     # ========== Search Filters ==========
#     st.subheader("Search Filters")
#     col1, col2 = st.columns(2)
#     with col1:
#         start_date = st.date_input(
#             "Start Date", 
#             value=datetime.date.today() - datetime.timedelta(days=30),
#             key="start_date_picker"
#         )
#     with col2:
#         end_date = st.date_input(
#             "End Date", 
#             value=datetime.date.today(),
#             key="end_date_picker"
#         )
    
#     keywords = st.text_input(
#         "Search Keywords (comma-separated)", 
#         help="Example: resume, CV, application",
#         key="keyword_search"
#     )

#     # ========== Analysis Button ==========
#     if st.button("🚀 Analyze Resumes", key="main_analyze_button"):
#         if not all([email_user, email_pass]) or not jd_text.strip():
#             st.error("❌ Please fill all required fields")
#             st.stop()
            
#         if '@' not in email_user or '.' not in email_user.split('@')[-1]:
#             st.error("❌ Invalid email address format")
#             st.stop()
            
#         with st.spinner("🔍 Scanning emails and analyzing resumes..."):
#             try:
#                 # Convert dates to IMAP format
#                 start_imap = start_date.strftime("%d-%b-%Y")
#                 end_imap = end_date.strftime("%d-%b-%Y")
                
#                 # Process keywords
#                 keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
                
#                 # Initialize processors
#                 doc_processor = DocumentProcessor()
#                 ai_processor = AIProcessor()
#                 semantic_analyzer = SemanticAnalyzer(jd_text)
                
#                 # Fetch emails with filters
#                 email_handler = EmailHandler(email_user, email_pass)
#                 attachments = email_handler.fetch_attachments(
#                     start_date=start_imap,
#                     end_date=end_imap,
#                     keywords=keyword_list,
#                     max_emails=50
#                 )
                
#                 candidates = []
#                 for attachment in attachments:
#                     try:
#                         # File validation
#                         ext = os.path.splitext(attachment['filename'])[1].lower()
#                         if ext not in ['.pdf', '.docx']:
#                             st.warning(f"Skipped {attachment['filename']} - Invalid file type")
#                             continue
                            
#                         # Process document
#                         docs = doc_processor.process_file(attachment['content'], ext)
#                         if len(docs[0].page_content) < 300:
#                             st.warning(f"Skipped {attachment['filename']} - Insufficient text")
#                             continue
                        
#                         # Run extraction chain
#                         # extraction_chain = ai_processor.create_extraction_chain()
#                         # result = extraction_chain.run({"input": docs[0].page_content})
        
#                         # # Handle results
#                         # if not result:
#                         #     continue
            
#                         # profile_data = result[0] if isinstance(result, list) else result
        
#                         # # Modified scoring
#                         # scoring_chain = ai_processor.create_scoring_chain(jd_text)
#                         # score_result = scoring_chain.run({
#                         #     "input": profile_data  # Key changed to 'input'
#                         # })
                        
#                         # # AI processing
#                         # extraction_chain = ai_processor.create_extraction_chain()
#                         # profile = extraction_chain.run({"resume_text": docs[0].page_content})
                        
#                         # # Scoring
#                         # scoring_chain = ai_processor.create_scoring_chain(jd_text)
#                         # score_result = scoring_chain.run({"candidate_profile": profile.json()})
                        
#                         # Extract profile once
#                         extraction_chain = ai_processor.create_extraction_chain()
#                         profile = extraction_chain.run({"resume_text": docs[0].page_content})

#                         # Score once using extracted profile
#                         scoring_chain = ai_processor.create_scoring_chain(jd_text)
#                         score_result = scoring_chain.run({"candidate_profile": profile.json()})

#                         # Semantic analysis
#                         similarity_score = semantic_analyzer.calculate_similarity(docs[0].page_content)
#                         final_score = 0.7*score_result.score + 0.3*similarity_score*100
                        
#                         # Store results
#                         candidates.append({
#                             **profile.dict(),
#                             **score_result.dict(),
#                             "similarity_score": similarity_score,
#                             "final_score": final_score,
#                             "download": create_download_link(
#                                 attachment['content'],
#                                 attachment['filename']
#                             )
#                         })
                        
#                     except Exception as e:
#                         st.error(f"⚠️ Error processing {attachment['filename']}: {str(e)}")
#                         continue
                
#                 if candidates:
#                     st.session_state.candidates = candidates
#                     st.session_state.show_results = True
#                     st.rerun()
#                 else:
#                     st.warning("🤷 No valid resumes found in attachments")
                    
#             except EmailFetchError as e:
#                 st.error(f"📧 Email error: {str(e)}")
#             except Exception as e:
#                 st.error(f"⚡ Unexpected error: {str(e)}")

# def results_page():
#     st.title("📊 AI Analysis Results")
    
#     if not st.session_state.get('candidates'):
#         st.warning("⚠️ No analysis results found. Please analyze resumes first.")
#         if st.button("🔙 Back to Main Page"):
#             st.session_state.show_results = False
#             st.rerun()
#         return

#     # Convert candidates to DataFrame
#     df = pd.DataFrame([{
#         "Name": c['name'],
#         "Score": c['final_score'],
#         "Skills": ", ".join(c['skills'][:5]),
#         "Experience": c['experience'],
#         "Education": ", ".join(c['education'][:2]),
#         "Certifications": len(c['certifications']),
#         "Semantic Match": f"{c['similarity_score']*100:.1f}%",
#         "Resume": c['download']
#     } for c in st.session_state.candidates])

#     # Sidebar Filters
#     with st.sidebar:
#         st.header("🔍 Filters")
#         min_score = st.slider("Minimum Score", 0, 100, 50)
#         required_skills = st.text_input("Required Skills (comma-separated)")
#         experience_range = st.slider("Experience Range (years)", 0, 50, (0, 50))

#     # Apply filters
#     filtered_df = df[
#         (df['Score'] >= min_score) &
#         (df['Experience'] >= experience_range[0]) &
#         (df['Experience'] <= experience_range[1])
#     ]
    
#     if required_skills:
#         skills_filter = [s.strip().lower() for s in required_skills.split(',')]
#         filtered_df = filtered_df[filtered_df['Skills'].str.lower().str.contains('|'.join(skills_filter))]

#     # Main Results Display
#     st.header("🧑💼 Ranked Candidates")
    
#     # Interactive Data Grid
#     st.data_editor(
#         filtered_df.sort_values("Score", ascending=False),
#         column_config={
#             "Score": st.column_config.ProgressColumn(
#                 format="%.1f%%",
#                 min_value=0,
#                 max_value=100,
#                 width="small"
#             ),
#             "Semantic Match": st.column_config.NumberColumn(
#                 format="%.1f%%",
#                 help="Semantic similarity to job description"
#             ),
#             "Resume": st.column_config.LinkColumn(
#                 display_text="📥 Download"
#             )
#         },
#         hide_index=True,
#         use_container_width=True
#     )

#     # Candidate Details Section
#     st.header("🔍 Candidate Details")
#     selected_name = st.selectbox(
#         "Select Candidate", 
#         filtered_df.sort_values("Score", ascending=False)['Name']
#     )
    
#     selected_candidate = next(
#         c for c in st.session_state.candidates 
#         if c['name'] == selected_name
#     )

#     with st.expander(f"📄 Full Analysis: {selected_name}"):
#         col1, col2, col3 = st.columns([1, 2, 1])
        
#         with col1:
#             st.metric("AI Match Score", f"{selected_candidate['final_score']:.1f}%")
#             st.metric("Experience", f"{selected_candidate['experience']} years")
#             st.markdown(selected_candidate['download'], unsafe_allow_html=True)
            
#         with col2:
#             st.subheader("🧠 AI Assessment")
#             st.write(selected_candidate['rationale'])
            
#             st.subheader("🚀 Improvement Suggestions")
#             for suggestion in selected_candidate['improvements']:
#                 st.markdown(f"- {suggestion}")
                
#         with col3:
#             st.subheader("📈 Match Breakdown")
#             st.metric("Semantic Similarity", 
#                      f"{selected_candidate['similarity_score']*100:.1f}%")
#             st.metric("Skill Matches", 
#                      f"{len(selected_candidate['skills'])}/{len(selected_candidate['skills'])}")
#             st.metric("Certifications", 
#                      f"{len(selected_candidate['certifications'])} found")

#     # Download Controls
#     st.divider()
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.download_button(
#             "📥 Download All Results (CSV)",
#             data=filtered_df.to_csv(index=False).encode('utf-8'),
#             file_name="cv_analysis_results.csv",
#             mime="text/csv"
#         )
    
#     with col2:
#         if st.button("🔁 New Analysis"):
#             st.session_state.clear()
#             st.rerun()

#     if st.button("🔙 Back to Main Page"):
#         st.session_state.show_results = False
#         st.rerun()

# if __name__ == "__main__":
#     # Initialize session state if not exists
#     if 'show_results' not in st.session_state:
#         st.session_state.show_results = False
    
#     # Single routing condition
#     if st.session_state.show_results:
#         results_page()
#     else:
#         main_page()


# # main.py
# import streamlit as st
# import pandas as pd
# from document_processor import DocumentProcessor
# from ai_processor import AIProcessor
# from semantic_analyzer import SemanticAnalyzer
# from utils import extract_text_from_pdf, extract_text_from_docx, create_download_link
# import os
# from config import Config
# from models import CandidateProfile
# import datetime

# def main_page():
#     st.title("📄 AI-Powered CV Analyzer")
    
#     # Date Range Selection
#     col1, col2 = st.columns(2)
#     with col1:
#         start_date = st.date_input("Start Date", 
#                                   value=datetime.date.today() - datetime.timedelta(days=30))
#     with col2:
#         end_date = st.date_input("End Date", 
#                                 value=datetime.date.today())
    
#     # Keyword Search
#     keywords = st.text_input("Search Keywords (comma-separated)", 
#                            help="Example: resume, CV, application")
    
#     # Email and processing inputs...
#     st.subheader("Email Credentials")
#     email_user = st.text_input("Email Address", help="Enter your email address")
#     email_pass = st.text_input("Email Password", type="password", help="Enter your email password")
    
#     if st.button("🚀 Analyze Resumes"):
#         if not all([email_user, email_pass, jd_text.strip()]):
#             st.error("❌ Please fill all required fields")
#             return
            
#         with st.spinner("🔍 Scanning emails and analyzing resumes..."):
#             try:
#                 # Convert dates to IMAP format
#                 start_imap = start_date.strftime("%d-%b-%Y")
#                 end_imap = end_date.strftime("%d-%b-%Y")
                
#                 # Process keywords
#                 keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
                
#                 # Fetch emails with date and keyword filters
#                 email_handler = EmailHandler(email_user, email_pass)
#                 attachments = email_handler.fetch_attachments(
#                     start_date=start_imap,
#                     end_date=end_imap,
#                     keywords=keyword_list
#                 )
#             except Exception as e:
#                 st.error(f"⚠️ Error fetching emails: {str(e)}")
#                 return
    
#     # Job Description Input
#     st.subheader("Job Requirements")
#     jd_text = st.text_area("Paste Job Description", height=200)
#     uploaded_jd = st.file_uploader("Upload Job Description (PDF/DOCX)", type=["pdf", "docx"])
    
#     # Process JD upload
#     if uploaded_jd:
#         file_content = uploaded_jd.read()
#         if uploaded_jd.name.endswith(".pdf"):
#             jd_text = extract_text_from_pdf(file_content)
#         else:
#             jd_text = extract_text_from_docx(file_content)
    
#     # Analysis Controls
#     if st.button("🚀 Analyze Resumes"):
#         if not all([email_user, email_pass, jd_text.strip()]):
#             st.error("❌ Please fill all required fields")
#             return
            
#         with st.spinner("🔍 Scanning emails and analyzing resumes..."):
#             try:
#                 # Initialize processors
#                 doc_processor = DocumentProcessor()
#                 ai_processor = AIProcessor()
#                 semantic_analyzer = SemanticAnalyzer(jd_text)
                
#                 # Fetch email attachments
#                 email_handler = EmailHandler(email_user, email_pass)
#                 attachments = email_handler.fetch_attachments()
                
#                 candidates = []
#                 for attachment in attachments:
#                     try:
#                         # Process document
#                         docs = doc_processor.process_file(attachment['content'], 
#                                                         os.path.splitext(attachment['filename'])[1])
                        
#                         # Extract candidate profile
#                         extraction_chain = ai_processor.create_extraction_chain()
#                         profile = extraction_chain.run({"resume_text": docs[0].page_content})
                        
#                         # Calculate scores
#                         scoring_chain = ai_processor.create_scoring_chain(jd_text)
#                         score_result = scoring_chain.run({"candidate_profile": profile.json()})
                        
#                         # Semantic analysis
#                         similarity_score = semantic_analyzer.calculate_similarity(docs[0].page_content)
#                         final_score = 0.7*score_result.score + 0.3*similarity_score*100
                        
#                         # Generate download link
#                         download_link = create_download_link(
#                             attachment['content'],
#                             attachment['filename']
#                         )
                        
#                         candidates.append({
#                             **profile.dict(),
#                             **score_result.dict(),
#                             "similarity_score": similarity_score,
#                             "final_score": final_score,
#                             "download": download_link
#                         })
                        
#                     except Exception as e:
#                         st.error(f"⚠️ Error processing {attachment['filename']}: {str(e)}")
#                         continue
                
#                 if candidates:
#                     st.session_state.candidates = candidates
#                     st.session_state.show_results = True
#                     st.rerun()
#                 else:
#                     st.warning("🤷 No valid resumes found in attachments")
                    
#             except EmailFetchError as e:
#                 st.error(f"📧 Email error: {str(e)}")
#             except Exception as e:
#                 st.error(f"⚡ Unexpected error: {str(e)}")

# # main.py
# # main.py
# import streamlit as st
# import pandas as pd
# import os
# import datetime
# from document_processor import DocumentProcessor
# from ai_processor import AIProcessor
# from semantic_analyzer import SemanticAnalyzer
# from email_handler import EmailHandler, EmailFetchError
# from utils import extract_text_from_pdf, extract_text_from_docx, create_download_link
# from config import Config
# from models import CandidateProfile

# def main_page():
#     st.title("📄 AI-Powered CV Analyzer")
    
#     # Initialize variables
#     jd_text = ""
    
#     # Job Description Section - MOVE TO TOP
#     st.subheader("Job Requirements")
#     jd_text = st.text_area("Paste Job Description", height=200, value="")
#     uploaded_jd = st.file_uploader("Upload Job Description (PDF/DOCX)", type=["pdf", "docx"])
    
#     # Process JD upload
#     if uploaded_jd:
#         file_content = uploaded_jd.read()
#         if uploaded_jd.name.endswith(".pdf"):
#             jd_text = extract_text_from_pdf(file_content)
#         else:
#             jd_text = extract_text_from_docx(file_content)

#     # Email Configuration
#     st.subheader("Email Credentials")
#     col1, col2 = st.columns(2)
#     with col1:
#         email_user = st.text_input("Email Address", help="Enter your email address")
#     with col2:
#         email_pass = st.text_input("Password", type="password", help="Enter your email password")

#     # Date Range Selection
#     st.subheader("Search Filters")
#     col1, col2 = st.columns(2)
#     with col1:
#         start_date = st.date_input("Start Date", 
#                                   value=datetime.date.today() - datetime.timedelta(days=30))
#     with col2:
#         end_date = st.date_input("End Date", 
#                                 value=datetime.date.today())
    
#     # Keyword Search
#     keywords = st.text_input("Search Keywords (comma-separated)", 
#                            help="Example: resume, CV, application")

#     # SINGLE Analyze Button with proper validation
#     if st.button("🚀 Analyze Resumes", key="analyze_button"):
#         if not all([email_user, email_pass]) or not jd_text.strip():
#             st.error("❌ Please fill all required fields")
#             return
            
#         with st.spinner("🔍 Scanning emails and analyzing resumes..."):
#             try:
#                 # Convert dates to IMAP format
#                 start_imap = start_date.strftime("%d-%b-%Y")
#                 end_imap = end_date.strftime("%d-%b-%Y")
                
#                 # Process keywords
#                 keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
                
#                 # Initialize processors
#                 doc_processor = DocumentProcessor()
#                 ai_processor = AIProcessor()
#                 semantic_analyzer = SemanticAnalyzer(jd_text)
                
#                 # Initialize email handler and fetch attachments
#                 email_handler = EmailHandler(email_user, email_pass)
#                 attachments = email_handler.fetch_attachments(
#                     start_date=start_imap,
#                     end_date=end_imap,
#                     keywords=keyword_list,
#                     max_emails=50
#                 )
                
#                 candidates = []
                
#                 for attachment in attachments:
#                     try:
#                         # File validation
#                         ext = os.path.splitext(attachment['filename'])[1].lower()
#                         if ext not in ['.pdf', '.docx']:
#                             st.warning(f"Skipped {attachment['filename']} - Invalid file type")
#                             continue
                            
#                         # Process document
#                         docs = doc_processor.process_file(attachment['content'], ext)
#                         if len(docs[0].page_content) < 300:
#                             st.warning(f"Skipped {attachment['filename']} - Insufficient text")
#                             continue
                            
#                         # AI processing
#                         extraction_chain = ai_processor.create_extraction_chain()
#                         profile = extraction_chain.run({"resume_text": docs[0].page_content})
                        
#                         # Scoring
#                         scoring_chain = ai_processor.create_scoring_chain(jd_text)
#                         score_result = scoring_chain.run({"candidate_profile": profile.json()})
                        
#                         # Semantic analysis
#                         similarity_score = semantic_analyzer.calculate_similarity(docs[0].page_content)
#                         final_score = 0.7*score_result.score + 0.3*similarity_score*100
                        
#                         # Store results
#                         candidates.append({
#                             **profile.dict(),
#                             **score_result.dict(),
#                             "similarity_score": similarity_score,
#                             "final_score": final_score,
#                             "download": create_download_link(
#                                 attachment['content'],
#                                 attachment['filename']
#                             )
#                         })
                        
#                     except Exception as e:
#                         st.error(f"⚠️ Error processing {attachment['filename']}: {str(e)}")
#                         continue
                
#                 if candidates:
#                     st.session_state.candidates = candidates
#                     st.session_state.show_results = True
#                     st.rerun()
#                 else:
#                     st.warning("🤷 No valid resumes found in attachments")
                    
#             except EmailFetchError as e:
#                 st.error(f"📧 Email error: {str(e)}")
#             except Exception as e:
#                 st.error(f"⚡ Unexpected error: {str(e)}")

# # Keep the results_page() and __main__ block unchanged from your original code


# # main.py
# import streamlit as st
# import pandas as pd
# from document_processor import DocumentProcessor
# from ai_processor import AIProcessor
# from semantic_analyzer import SemanticAnalyzer
# from utils import extract_text_from_pdf, extract_text_from_docx, create_download_link
# from email_handler import EmailHandler, EmailFetchError  
# import os
# from config import Config
# from models import CandidateProfile
# import datetime

# def main_page():
#     st.title("📄 AI-Powered CV Analyzer")
#     if st.checkbox("Using Corporate Network"):
#         os.environ["USE_CORPORATE_NETWORK"] = "True"
        
#     # Initialize variables first
#     jd_text = ""
    
#     # ========== Job Description Section ========== 
#     st.subheader("Job Requirements")
    
#     # Text input with unique key
#     jd_text = st.text_area(
#         "Paste Job Description", 
#         height=200, 
#         value="",
#         key="main_job_description_text"
#     )
    
#     # File uploader with unique key
#     uploaded_jd = st.file_uploader(
#         "Upload Job Description (PDF/DOCX/TXT)",
#         type=["pdf", "docx", "txt"],
#         key="job_description_upload_widget"
#     )
    
#     # Process JD upload
#     if uploaded_jd:
#         file_content = uploaded_jd.read()
#         if uploaded_jd.name.endswith(".pdf"):
#             jd_text = extract_text_from_pdf(file_content)
#         elif uploaded_jd.name.endswith(".docx"):
#             jd_text = extract_text_from_docx(file_content)
#         elif uploaded_jd.name.endswith(".txt"):
#             jd_text = file_content.decode("utf-8")
    

#     # ========== Connection Troubleshooting ==========
#     with st.expander("🔧 Connection Troubleshooting Guide"):
#         st.markdown("""
#         **Common Fixes:**
#         1. Enable IMAP in your email account settings
#         2. For Gmail: [Enable IMAP](https://mail.google.com/mail/#settings/fwdandpop)
#         3. Use App Password if 2FA enabled
#         4. Check firewall/antivirus isn't blocking port 993
#         5. Try different network (e.g., switch from WiFi to mobile hotspot)
#         """)

#     # ========== Email Credentials ==========
#     st.subheader("Email Credentials")
#     col1, col2 = st.columns(2)
#     with col1:
#         email_user = st.text_input("Email Address", key="email_input")
#     with col2:
#         email_pass = st.text_input("Password", type="password", key="password_input")

#     # ========== Search Filters ==========
#     st.subheader("Search Filters")
#     col1, col2 = st.columns(2)
#     with col1:
#         start_date = st.date_input(
#             "Start Date", 
#             value=datetime.date.today() - datetime.timedelta(days=30),
#             key="start_date_picker"
#         )
#     with col2:
#         end_date = st.date_input(
#             "End Date", 
#             value=datetime.date.today(),
#             key="end_date_picker"
#         )
    
#     keywords = st.text_input(
#         "Search Keywords (comma-separated)", 
#         help="Example: resume, CV, application",
#         key="keyword_search"
#     )

#     # ========== Analysis Button ==========
#     if st.button("🚀 Analyze Resumes", key="main_analyze_button"):
#         if not all([email_user, email_pass]) or not jd_text.strip():
#             st.error("❌ Please fill all required fields")
#             st.stop()
            
#         if '@' not in email_user or '.' not in email_user.split('@')[-1]:
#             st.error("❌ Invalid email address format")
#             st.stop()
            
#         with st.spinner("🔍 Scanning emails and analyzing resumes..."):
#             try:
#                 # Convert dates to IMAP format
#                 start_imap = start_date.strftime("%d-%b-%Y")
#                 end_imap = end_date.strftime("%d-%b-%Y")
                
#                 # Process keywords
#                 keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
                
#                 # Initialize processors
#                 doc_processor = DocumentProcessor()
#                 ai_processor = AIProcessor()
#                 semantic_analyzer = SemanticAnalyzer(jd_text)
                
#                 # Fetch emails with filters
#                 email_handler = EmailHandler(email_user, email_pass)
#                 attachments = email_handler.fetch_attachments(
#                     start_date=start_imap,
#                     end_date=end_imap,
#                     keywords=keyword_list,
#                     max_emails=50
#                 )
                
#                 candidates = []
#                 for attachment in attachments:
                    
#                     try:
#                         # File validation
#                         ext = os.path.splitext(attachment['filename'])[1].lower()
#                         if ext not in ['.pdf', '.docx']:
#                             st.warning(f"Skipped {attachment['filename']} - Invalid file type")
#                             continue
                            
#                         # Process document
#                         docs = doc_processor.process_file(attachment['content'], ext)
#                         if len(docs[0].page_content) < 300:
#                             st.warning(f"Skipped {attachment['filename']} - Insufficient text")
#                             continue
#                         # Extract profile once
#                         extraction_chain = ai_processor.create_extraction_chain()
#                         profile = extraction_chain.run({"resume_text": docs[0].page_content})

#                         # Score once using extracted profile
#                         scoring_chain = ai_processor.create_scoring_chain(jd_text)
#                         score_result = scoring_chain.run({"candidate_profile": profile})
                        

#                         # Semantic analysis
#                         similarity_score = semantic_analyzer.calculate_similarity(docs[0].page_content)
#                         # final_score = 0.7*score_result.score + 0.3*similarity_score*100
#                         final_score = 0.7 * score_result['score'] + 0.3 * similarity_score * 100
                        
#                         # Store results
#                         candidates.append({
#                             **profile,
#                             **score_result,
#                             "similarity_score": similarity_score,
#                             "final_score": final_score,
#                             "download": create_download_link(
#                                 attachment['content'],
#                                 attachment['filename']
#                             )
#                         })
                        
#                     except Exception as e:
#                         st.error(f"⚠️ Error processing {attachment['filename']}: {str(e)}")
#                         continue
                
#                 if candidates:
#                     st.session_state.candidates = candidates
#                     st.session_state.show_results = True
#                     st.rerun()
#                 else:
#                     st.warning("🤷 No valid resumes found in attachments")
                    
#             except EmailFetchError as e:
#                 st.error(f"📧 Email error: {str(e)}")
#             except Exception as e:
#                 st.error(f"⚡ Unexpected error: {str(e)}")

# def results_page():
#     st.title("📊 AI Analysis Results")
    
#     if not st.session_state.get('candidates'):
#         st.warning("⚠️ No analysis results found. Please analyze resumes first.")
#         if st.button("🔙 Back to Main Page"):
#             st.session_state.show_results = False
#             st.rerun()
#         return

#     # Convert candidates to DataFrame
#     df = pd.DataFrame([{
#         "Name": c['name'],
#         "Score": c['final_score'],
#         "Skills": ", ".join(c['skills'][:5]),
#         "Experience": c['experience'],
#         "Education": ", ".join(c['education'][:2]),
#         "Certifications": len(c['certifications']),
#         "Semantic Match": f"{c['similarity_score']*100:.1f}%",
#         "Resume": c['download']
#     } for c in st.session_state.candidates])

#     # Sidebar Filters
#     with st.sidebar:
#         st.header("🔍 Filters")
#         min_score = st.slider("Minimum Score", 0, 100, 50)
#         required_skills = st.text_input("Required Skills (comma-separated)")
#         experience_range = st.slider("Experience Range (years)", 0, 50, (0, 50))

#     # Apply filters
#     filtered_df = df[
#         (df['Score'] >= min_score) &
#         (df['Experience'] >= experience_range[0]) &
#         (df['Experience'] <= experience_range[1])
#     ]
    
#     if required_skills:
#         skills_filter = [s.strip().lower() for s in required_skills.split(',')]
#         filtered_df = filtered_df[filtered_df['Skills'].str.lower().str.contains('|'.join(skills_filter))]
        

#     # Main Results Display
#     st.header("🧑💼 Ranked Candidates")
    
#     # Interactive Data Grid
#     st.data_editor(
#         filtered_df.sort_values("Score", ascending=False),
#         column_config={
#             "Score": st.column_config.ProgressColumn(
#                 format="%.1f%%",
#                 min_value=0,
#                 max_value=100,
#                 width="small"
#             ),
#             "Semantic Match": st.column_config.NumberColumn(
#                 format="%.1f%%",
#                 help="Semantic similarity to job description"
#             ),
#             "Resume": st.column_config.LinkColumn(
#                 display_text="📥 Download"
#             )
#         },
#         hide_index=True,
#         use_container_width=True
#     )

#     # Candidate Details Section
#     st.header("🔍 Candidate Details")
#     selected_name = st.selectbox(
#         "Select Candidate", 
#         filtered_df.sort_values("Score", ascending=False)['Name']
#     )

#     # selected_candidate = next(
#     #     c for c in st.session_state.candidates 
#     #     if c['name'] == selected_name
#     # )
#     selected_candidate = next(
#     (c for c in st.session_state.candidates if c['name'] == selected_name), 
#     None
# )



#     with st.expander(f"📄 Full Analysis: {selected_name}"):
#         col1, col2, col3 = st.columns([1, 2, 1])
        
#         with col1:
#             st.metric("AI Match Score", f"{selected_candidate['final_score']:.1f}%")
#             st.metric("Experience", f"{selected_candidate['experience']} years")
#             st.markdown(selected_candidate['download'], unsafe_allow_html=True)
            
#         with col2:
#             st.subheader("🧠 AI Assessment")
#             st.write(selected_candidate['rationale'])
            
#             st.subheader("🚀 Improvement Suggestions")
#             for suggestion in selected_candidate['improvements']:
#                 st.markdown(f"- {suggestion}")
                
#         with col3:
#             st.subheader("📈 Match Breakdown")
#             st.metric("Semantic Similarity", 
#                      f"{selected_candidate['similarity_score']*100:.1f}%")
#             st.metric("Skill Matches", 
#                      f"{len(selected_candidate['skills'])}/{len(selected_candidate['skills'])}")
#             st.write(", ".join(selected_candidate['skills']))

#             st.metric("Certifications", 
#                      f"{len(selected_candidate['certifications'])} found")
#             st.write(", ".join(selected_candidate['certifications']))

#     # Download Controls
#     st.divider()
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.download_button(
#             "📥 Download All Results (CSV)",
#             data=filtered_df.to_csv(index=False).encode('utf-8'),
#             file_name="cv_analysis_results.csv",
#             mime="text/csv"
#         )
    
#     with col2:
#         if st.button("🔁 New Analysis"):
#             st.session_state.clear()
#             st.rerun()

#     if st.button("🔙 Back to Main Page"):
#         st.session_state.show_results = False
# #         st.rerun()

# if __name__ == "__main__":
#     # Initialize session state if not exists
#     if 'show_results' not in st.session_state:
#         st.session_state.show_results = False
    
#     # Single routing condition
#     if st.session_state.show_results:
#         results_page()
#     else:
#         main_page()
# 
# # main.py
import streamlit as st
import pandas as pd
from document_processor import DocumentProcessor
from ai_processor import AIProcessor
from semantic_analyzer import SemanticAnalyzer
from utils import extract_text_from_pdf, extract_text_from_docx, create_download_link
from email_handler import EmailHandler, EmailFetchError  
import os
from config import Config
from models import CandidateProfile
import datetime

def job_desc_page():
    # Page Navigation Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("📄 AI-Powered CV Analyzer")
    with col2:
        if st.button("⚙️ Email Config", key="nav_email_config"):
            st.session_state.current_page = 'email_config'
            st.rerun()

    # Initialize variables
    jd_text = ""

    # ========== Job Description Section ========== 
    st.subheader("Job Requirements")
    
    # Text input
    jd_text = st.text_area(
        "Paste Job Description", 
        height=200, 
        value="",
        key="main_job_description_text"
    )
    
    # File uploader
    uploaded_jd = st.file_uploader(
        "Upload Job Description (PDF/DOCX/TXT)",
        type=["pdf", "docx", "txt"],
        key="job_description_upload_widget"
    )
    
    # Process JD upload
    if uploaded_jd:
        file_content = uploaded_jd.read()
        if uploaded_jd.name.endswith(".pdf"):
            jd_text = extract_text_from_pdf(file_content)
        elif uploaded_jd.name.endswith(".docx"):
            jd_text = extract_text_from_docx(file_content)
        elif uploaded_jd.name.endswith(".txt"):
            jd_text = file_content.decode("utf-8")

    # Next Page Button
    if st.button("Next →", key="job_desc_next"):
        if not jd_text.strip():
            st.error("❌ Please enter a job description")
        else:
            st.session_state.jd_text = jd_text
            st.session_state.current_page = 'email_config'
            st.rerun()

def email_config_page():
    # Page Navigation Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("📧 Email Configuration")
    with col2:
        if st.button("← Job Description", key="nav_job_desc"):
            st.session_state.current_page = 'job_desc'
            st.rerun()

    # ========== Network Configuration ==========
    if st.checkbox("Using Corporate Network", key="corporate_network"):
        os.environ["USE_CORPORATE_NETWORK"] = "True"

    # ========== Email Credentials ==========
    st.subheader("Email Credentials")
    col1, col2 = st.columns(2)
    with col1:
        email_user = st.text_input("Email Address", key="email_input")
    with col2:
        email_pass = st.text_input("Password", type="password", key="password_input")

    # ========== Search Filters ==========
    st.subheader("Search Filters")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date", 
            value=datetime.date.today() - datetime.timedelta(days=30),
            key="start_date_picker"
        )
    with col2:
        end_date = st.date_input(
            "End Date", 
            value=datetime.date.today(),
            key="end_date_picker"
        )
    
    keywords = st.text_input(
        "Search Keywords (comma-separated)", 
        help="Example: resume, CV, application",
        key="keyword_search"
    )

    # ========== Connection Troubleshooting ==========
    with st.expander("🔧 Connection Troubleshooting Guide"):
        st.markdown("""
        **Common Fixes:**
        1. Enable IMAP in your email account settings
        2. For Gmail: [Enable IMAP](https://mail.google.com/mail/#settings/fwdandpop)
        3. Use App Password if 2FA enabled
        4. Check firewall/antivirus isn't blocking port 993
        5. Try different network (e.g., switch from WiFi to mobile hotspot)
        """)

    # ========== Analysis Button ==========
    if st.button("🚀 Analyze Resumes", key="email_analyze_button"):
        if not all([email_user, email_pass]):
            st.error("❌ Please fill email credentials")
            st.stop()
            
        if '@' not in email_user or '.' not in email_user.split('@')[-1]:
            st.error("❌ Invalid email address format")
            st.stop()

        if 'jd_text' not in st.session_state or not st.session_state.jd_text.strip():
            st.error("❌ Missing job description! Return to main page")
            st.stop()
            
        with st.spinner("🔍 Scanning emails and analyzing resumes..."):
            try:
                # Convert dates to IMAP format
                start_imap = start_date.strftime("%d-%b-%Y")
                end_imap = end_date.strftime("%d-%b-%Y")
                
                # Process keywords
                keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
                
                # Initialize processors
                doc_processor = DocumentProcessor()
                ai_processor = AIProcessor()
                semantic_analyzer = SemanticAnalyzer(st.session_state.jd_text)
                
                # Fetch emails with filters
                email_handler = EmailHandler(email_user, email_pass)
                attachments = email_handler.fetch_attachments(
                    start_date=start_imap,
                    end_date=end_imap,
                    keywords=keyword_list,
                    max_emails=50
                )
                
                candidates = []
                for attachment in attachments:                    
                    try:
                        # File validation
                        ext = os.path.splitext(attachment['filename'])[1].lower()
                        if ext not in ['.pdf', '.docx']:
                            st.warning(f"Skipped {attachment['filename']} - Invalid file type")
                            continue
                            
                        # Process document
                        docs = doc_processor.process_file(attachment['content'], ext)
                        if len(docs[0].page_content) < 300:
                            st.warning(f"Skipped {attachment['filename']} - Insufficient text")
                            continue
                            
                        # Extract profile once
                        extraction_chain = ai_processor.create_extraction_chain()
                        profile = extraction_chain.run({"resume_text": docs[0].page_content})

                        # Score once using extracted profile
                        scoring_chain = ai_processor.create_scoring_chain(st.session_state.jd_text)
                        score_result = scoring_chain.run({"candidate_profile": profile})
                        
                        # Semantic analysis
                        similarity_score = semantic_analyzer.calculate_similarity(docs[0].page_content)
                        final_score = 0.7 * score_result['score'] + 0.3 * similarity_score * 100
                        
                        # Store results
                        candidates.append({
                            **profile,
                            **score_result,
                            "similarity_score": similarity_score,
                            "final_score": final_score,
                            "download": create_download_link(
                                attachment['content'],
                                attachment['filename']
                            )
                        })
                        
                    except Exception as e:
                        st.error(f"⚠️ Error processing {attachment['filename']}: {str(e)}")
                        continue
                
                if candidates:
                    st.session_state.candidates = candidates
                    st.session_state.show_results = True
                    st.rerun()
                else:
                    st.warning("🤷 No valid resumes found in attachments")
                    
            except EmailFetchError as e:
                st.error(f"📧 Email error: {str(e)}")
            except Exception as e:
                st.error(f"⚡ Unexpected error: {str(e)}")

def results_page():
    st.title("📊 AI Analysis Results")
    
    if not st.session_state.get('candidates'):
        st.warning("⚠️ No analysis results found. Please analyze resumes first.")
        if st.button("🔙 Back to Main Page", key="back_from_empty"):
            st.session_state.show_results = False
            st.rerun()
        return

    # Convert candidates to DataFrame
    df = pd.DataFrame([{
        "Name": c['name'],
        "Score": c['final_score'],
        "Skills": ", ".join(c['skills'][:5]),
        "Experience": c['experience'],
        "Education": ", ".join(c['education'][:2]),
        "Certifications": len(c['certifications']),
        "Semantic Match": f"{c['similarity_score']*100:.1f}%",
        "Resume": c['download']
    } for c in st.session_state.candidates])

    # Sidebar Filters
    with st.sidebar:
        st.header("🔍 Filters")
        min_score = st.slider("Minimum Score", 0, 100, 50, key="min_score_filter")
        required_skills = st.text_input("Required Skills (comma-separated)", key="required_skills")
        experience_range = st.slider("Experience Range (years)", 0, 50, (0, 50), key="exp_range")

    # Apply filters
    filtered_df = df[
        (df['Score'] >= min_score) &
        (df['Experience'] >= experience_range[0]) &
        (df['Experience'] <= experience_range[1])
    ]
    
    if required_skills:
        skills_filter = [s.strip().lower() for s in required_skills.split(',')]
        filtered_df = filtered_df[filtered_df['Skills'].str.lower().str.contains('|'.join(skills_filter))]
        

    # Main Results Display
    st.header("🧑💼 Ranked Candidates")
    
    # Interactive Data Grid
    st.data_editor(
        filtered_df.sort_values("Score", ascending=False),
        column_config={
            "Score": st.column_config.ProgressColumn(
                format="%.1f%%",
                min_value=0,
                max_value=100,
                width="small"
            ),
            "Semantic Match": st.column_config.NumberColumn(
                format="%.1f%%",
                help="Semantic similarity to job description"
            ),
            "Resume": st.column_config.LinkColumn(
                display_text="📥 Download"
            )
        },
        hide_index=True,
        use_container_width=True,
        key="results_grid"
    )

    # Candidate Details Section
    st.header("🔍 Candidate Details")
    selected_name = st.selectbox(
        "Select Candidate", 
        filtered_df.sort_values("Score", ascending=False)['Name'],
        key="candidate_select"
    )

    selected_candidate = next(
        (c for c in st.session_state.candidates if c['name'] == selected_name), 
        None
    )

    with st.expander(f"📄 Full Analysis: {selected_name}"):
        st.write("Candidate details here...")

        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.metric("AI Match Score", f"{selected_candidate['final_score']:.1f}%")
            st.metric("Experience", f"{selected_candidate['experience']} years")
            st.markdown(selected_candidate['download'], unsafe_allow_html=True)
            
        with col2:
            st.subheader("🧠 AI Assessment")
            st.write(selected_candidate['rationale'])
            
            st.subheader("🚀 Improvement Suggestions")
            for suggestion in selected_candidate['improvements']:
                st.markdown(f"- {suggestion}")
                
        with col3:
            st.subheader("📈 Match Breakdown")
            st.metric("Semantic Similarity", 
                     f"{selected_candidate['similarity_score']*100:.1f}%")
            st.metric("Skill Matches", 
                     f"{len(selected_candidate['skills'])}/{len(selected_candidate['skills'])}")
            st.write(", ".join(selected_candidate['skills']))

            st.metric("Certifications", 
                     f"{len(selected_candidate['certifications'])} found")
            st.write(", ".join(selected_candidate['certifications']))

    # Download Controls
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            "📥 Download All Results (CSV)",
            data=filtered_df.to_csv(index=False).encode('utf-8'),
            file_name="cv_analysis_results.csv",
            mime="text/csv",
            key="download_csv"
        )
    
    with col2:
        if st.button("🔁 New Analysis", key="new_analysis"):
            st.session_state.clear()
            st.rerun()

    if st.button("🔙 Back to Main Page", key="back_main"):
        st.session_state.show_results = False
        st.rerun()

if __name__ == "__main__":
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'job_desc'
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    
    # Page routing
    if st.session_state.show_results:
        results_page()
    elif st.session_state.current_page == 'job_desc':
        job_desc_page()
    elif st.session_state.current_page == 'email_config':
        email_config_page()
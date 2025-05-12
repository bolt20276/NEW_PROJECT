# import numpy as np
# from langchain_community.vectorstores import FAISS
# from langchain.embeddings import OpenAIEmbeddings
# from config import Config
# from typing import List

# class SemanticAnalyzer:
#     def __init__(self, job_description: str):
#         self.embeddings = OpenAIEmbeddings(
#             model=Config.EMBEDDING_MODEL,
#             api_key=Config.OPENAI_API_KEY
#         )
#         self.jd_store = self._create_store([job_description])
        
#     def _create_store(self, texts: List[str]):
#         return FAISS.from_texts(texts, self.embeddings)
    
#     def calculate_similarity(self, resume_text: str):
#         resume_store = self._create_store([resume_text])
#         jd_embeddings = self.jd_store.index.reconstruct_n(0, self.jd_store.index.ntotal)
#         cv_embeddings = resume_store.index.reconstruct_n(0, resume_store.index.ntotal)
        
#         jd_norm = jd_embeddings / np.linalg.norm(jd_embeddings, axis=1, keepdims=True)
#         cv_norm = cv_embeddings / np.linalg.norm(cv_embeddings, axis=1, keepdims=True)
        
#         return np.dot(jd_norm, cv_norm.T).max(axis=0).mean()

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from config import Config
from typing import List

class SemanticAnalyzer:
    def __init__(self, job_description: str):
        self.embeddings = OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            api_key=Config.OPENAI_API_KEY
        )
        self.jd_store = self._create_store([job_description])
        
    def _create_store(self, texts: List[str]):
        return FAISS.from_texts(texts, self.embeddings)
    
    def calculate_similarity(self, resume_text: str):
        resume_store = self._create_store([resume_text])
        jd_embeddings = self.jd_store.index.reconstruct_n(0, self.jd_store.index.ntotal)
        cv_embeddings = resume_store.index.reconstruct_n(0, resume_store.index.ntotal)
        
        jd_norm = jd_embeddings / np.linalg.norm(jd_embeddings, axis=1, keepdims=True)
        cv_norm = cv_embeddings / np.linalg.norm(cv_embeddings, axis=1, keepdims=True)
        
        return np.dot(jd_norm, cv_norm.T).max(axis=0).mean()
# # ai_processor.py
# from langchain.chains import LLMChain  # ✅ Correct import path
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import JsonOutputParser
# from models import CandidateProfile, ScoringResult
# import os

# class Config:
#     LLM_MODEL = "gpt-3.5-turbo"
#     TEMPERATURE = 0.7
#     OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Store in .env file

# class AIProcessor:
#     def __init__(self):
#         self.llm = ChatOpenAI(
#             model=Config.LLM_MODEL,
#             temperature=Config.TEMPERATURE,
#             api_key=Config.OPENAI_API_KEY
#         )

#     def create_extraction_chain(self):
#         parser = JsonOutputParser(pydantic_object=CandidateProfile)
#         prompt = ChatPromptTemplate.from_template(
#             "Extract structured information from resume:\n{format_instructions}\nResume Content: {resume_text}",
#             input_variables=["resume_text"]
#         )
#         return LLMChain(
#             llm=self.llm,
#             prompt=prompt.partial(format_instructions=parser.get_format_instructions()),
#             output_parser=parser
#         )

#     def create_scoring_chain(self, job_description: str):
#         parser = JsonOutputParser(pydantic_object=ScoringResult)
#         prompt = ChatPromptTemplate.from_template(
#             "Score candidate against job description:\n{format_instructions}\nJob Description: {job_description}\nCandidate Profile: {candidate_profile}",
#             input_variables=["candidate_profile", "job_description"]
#         )
#         return LLMChain(
#             llm=self.llm,
#             prompt=prompt.partial(
#                 format_instructions=parser.get_format_instructions(),
#                 job_description=job_description
#             ),
#             output_parser=parser
#         )

# ai_processor.py
from langchain.chains import LLMChain  # ✅ Correct import path
# from langchain_openai import ChatOpenAI
from langchain.chat_models import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from models import CandidateProfile, ScoringResult
import os

class Config:
    LLM_MODEL = "gpt-3.5-turbo"
    TEMPERATURE = 0.7
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Store in .env file

class AIProcessor:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=Config.TEMPERATURE,
            api_key=Config.OPENAI_API_KEY
        )

    def create_extraction_chain(self):
        parser = JsonOutputParser(pydantic_object=CandidateProfile)
        prompt = ChatPromptTemplate.from_template(
            "Extract structured information from resume:\n{format_instructions}\nResume Content: {resume_text}",
            # input_variables=["resume_text"]
        )
        return LLMChain(
            llm=self.llm,
            prompt=prompt.partial(format_instructions=parser.get_format_instructions()),
            output_parser=parser
        )

    def create_scoring_chain(self, job_description: str):
        parser = JsonOutputParser(pydantic_object=ScoringResult)
        prompt = ChatPromptTemplate.from_template(
            "Score candidate against job description:\n{format_instructions}\nJob Description: {job_description}\nCandidate Profile: {candidate_profile}",
            # input_variables=["candidate_profile", "job_description"]
        )
        return LLMChain(
            llm=self.llm,
            prompt=prompt.partial(
                format_instructions=parser.get_format_instructions(),
                job_description=job_description
            ),
            output_parser=parser
        )
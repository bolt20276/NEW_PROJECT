from pydantic import BaseModel, Field
from typing import List

class CandidateProfile(BaseModel):
    name: str = Field(description="Full name of the candidate")
    email: str = Field(description="Email address")
    skills: List[str] = Field(description="List of technical skills")
    education: List[str] = Field(description="Educational qualifications")
    experience: float = Field(description="Total years of experience")
    certifications: List[str] = Field(description="Professional certifications")
    responsibilities: List[str] = Field(description="Key responsibilities from resume")

class ScoringResult(BaseModel):
    score: float = Field(description="Overall match score 0-100")
    rationale: str = Field(description="Scoring explanation")
    improvements: List[str] = Field(description="Suggested improvements")
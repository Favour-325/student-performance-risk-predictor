from pydantic import BaseModel, Field
from typing import Literal

class StudentInput(BaseModel):
    G1: int = Field(..., ge=0, le=20, description="First period grade")
    G2: int = Field(..., ge=0, le=20, description="Second period grade")
    studytime: int = Field(..., ge=1, le=4, description="Weekly study time (1–4)")
    absences: int = Field(..., ge=0, le=93, description="Number of absences")
    age: int = Field(..., ge=15, description="Student's age")
    schoolsup:  Literal['yes', 'no']
    Medu: int = Field(..., ge=0, le=4, description="Mother's Education")
    Fedu: int = Field(..., ge=0, le=4, description="Father's Education")
    traveltime: int = Field(..., ge=0, le=4, description="Travel time between home and school")
    Walc: int = Field(..., ge=1, le=5, description="Work day alcohol consumption")
    Dalc: int = Field(..., ge=1, le=5, description="Weekend alcohol consumption")
    failures: int = Field(..., ge=1, le=4, description="Number of past class failures")
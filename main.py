from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import joblib

#load model
model=joblib.load("/workspaces/LoanPredict/loan_classifier.joblib")

app=FastAPI()

class Input(BaseModel):
    install:int
    
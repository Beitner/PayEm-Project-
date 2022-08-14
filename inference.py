from catboost import CatBoostClassifier, EShapCalcType, EFeaturesSelectionAlgorithm, Pool
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import uvicorn
from fastapi import FastAPI, Form, Depends, Path
import pickle
from typing import Union, Optional, List
from pydantic import BaseModel
from locale import currency
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy import Boolean, Column, Float, String, Integer
from prediction import prediction
from train1 import train1

app = FastAPI()


class Course(BaseModel):
    Request_title: str
    Request_decription: str
    Request_amount: float
    currency: Optional[str] = None
    Expense_date: str
    Expense_category: Optional[str] = None
    Sub_company: str
    Department: str


data = pd.read_csv('data_translated.csv', parse_dates=['created'])
model, cat_features, num_features, upper_threshold, lower_threshold, bertopic_model = train1(
    data)


# Reading parameters from the user:
@app.post("/predict/")
def create_course(course: Course):
    print("Input Data from user:")
    print(course.Request_title)
    print(course.Request_decription)
    print(course.Request_amount)
    print(course.currency)
    print(course.Expense_date)
    print(course.Expense_category)
    print(course.Sub_company)
    print(course.Department)
    prediction(model, cat_features, num_features, upper_threshold, lower_threshold, bertopic_model, course.Request_amount, course.currency,
               course.Expense_date, course.Expense_category, course.Sub_company, course.Request_title, course.Request_decription)

    return course


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

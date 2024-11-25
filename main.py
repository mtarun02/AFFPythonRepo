from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json
import uvicorn
from pydantic import create_model
from pycaret.classification import load_model, predict_model
import pandas as pd
from typing import Union
import logging

app = FastAPI()

# logging is a more stable tool for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# An example: logger.info({Something you want to print out})

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from all origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class FormData(BaseModel):
    # Input fields in the required order
    fieldPamidronate: str
    fieldRisedronate: str
    fieldIbandronate: str
    fieldZoledronate: str
    fieldAlendronate: str
    fieldDenosumab: str
    fieldDuration: str
    fieldPreviousNTF: str
    fieldParent: str
    fieldSmoker: str
    fieldSteroid: str
    fieldRA: str
    fieldSO: str
    fieldAlcoholism: str
    fieldMCT: str
    fieldLCT: str
    fieldOffset: str
    fieldFMA: str
    fieldFMD: str
    fieldVO: str
    fieldFMT: str
    fieldDXABMD: str
    fieldDXALumbar: str
    fieldFraxMajor: str
    fieldFraxHip: str
    
    
def convert_to_nan(value):
    """
    Convert empty strings to 'nan' for fields that need numeric inputs.
    """
    if value == "":
        return "nan"
    return value


#load model and create it 
model = load_model("AFF_Prediction_ET")
input_model = create_model


def predict(data):
    """
    Function to predict using the loaded model.
    It converts empty strings to 'nan' for numerical fields.
    """

    # Convert fields to 'nan' if they are empty
    data = [convert_to_nan(val) for val in data]

    # Create DataFrame for the input data
    df = pd.DataFrame([data], columns=[
        'Pamidronate Use', 'Risedronate Use', 'Ibandronate Yse', 'Zoledronate Use',
        'Alendronate Use', 'Duration of Bisphosphonate Use', 'Denosumab Use',
        'Previous non traumatic fracture', 'Parent fractured hip (mother or father)',
        'Current smoker', 'Steroid Use (> 3 months)', 'Rheumatoid arthritis',
        'History of secondary osteoporosis', 'Alcoholism (> 3 drinks per day)',
        'Medial cortical thickness at 50 mm', 'Lateral cortical thickness at 50 mm',
        'Femoral Horizontal Offset', 'Femoral neck angle', 'Femural Head Diameter',
        'Femoral Vertical Offset', 'Femoral neck thickness',
        'DXA Femoral Neck #1 BMD (g/cm^2)', 'DXA L1-L4 BMD (g/cm^2)',
        'FRAX Score - online calculator major fracture (%)',
        'FRAX Score - online calculator hip fracture (%)'
    ])

    # Make predictions using the loaded model
    predictions = predict_model(model, data=df)

    return {
        'Prediction': str(predictions["prediction_label"].iloc[0]),
        'Probability of Prediction': float(predictions["prediction_score"].iloc[0])
    }


@app.post("/submit/")
async def submit_form(data: FormData):
    try:
        # Get the form data as a dictionary
        form_data = data.dict()

        # Create form_data_list in the required order using form_data.get()
        form_data_list = [
            convert_to_nan(form_data.get('fieldPamidronate')),
            convert_to_nan(form_data.get('fieldRisedronate')),
            convert_to_nan(form_data.get('fieldIbandronate')),
            convert_to_nan(form_data.get('fieldZoledronate')),
            convert_to_nan(form_data.get('fieldAlendronate')),
            convert_to_nan(form_data.get('fieldDuration')),
            convert_to_nan(form_data.get('fieldDenosumab')),         
            convert_to_nan(form_data.get('fieldPreviousNTF')),
            convert_to_nan(form_data.get('fieldParent')),
            convert_to_nan(form_data.get('fieldSmoker')),
            convert_to_nan(form_data.get('fieldSteroid')),
            convert_to_nan(form_data.get('fieldRA')),
            convert_to_nan(form_data.get('fieldSO')),
            convert_to_nan(form_data.get('fieldAlcoholism')),
            convert_to_nan(form_data.get('fieldMCT')),
            convert_to_nan(form_data.get('fieldLCT')),
            convert_to_nan(form_data.get('fieldOffset')),
            convert_to_nan(form_data.get('fieldFMA')),
            convert_to_nan(form_data.get('fieldFMD')),
            convert_to_nan(form_data.get('fieldVO')),
            convert_to_nan(form_data.get('fieldFMT')),
            convert_to_nan(form_data.get('fieldDXABMD')),
            convert_to_nan(form_data.get('fieldDXALumbar')),
            convert_to_nan(form_data.get('fieldFraxMajor')),
            convert_to_nan(form_data.get('fieldFraxHip'))
        ]

        # Call the predict function with the ordered data
        AFF_prediction = predict(form_data_list)

        # Return prediction as JSON
        return JSONResponse(content={"Prediction": AFF_prediction["Prediction"], 'Probability': AFF_prediction["Probability of Prediction"]})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

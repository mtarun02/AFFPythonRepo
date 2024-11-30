# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("AFF_Prediction_ET_T_scores")

# Create input/output pydantic models
input_model = create_model("AFF_Prediction_ET_T_scores_input", **{'Pamidronate Use': 'N', 'Risedronate Use': 'N', 'Ibandronate Yse': 'N', 'Zoledronate Use': 'N', 'Alendronate Use': 'Y', 'Duration of Bisphosphonate Use': 10.0, 'Denosumab Use': 'N', 'Previous non traumatic fracture': 'Y', 'Parent fractured hip (mother or father)': 'N', 'Current smoker': 'N', 'Steroid Use (> 3 months)': 'Y', 'Rheumatoid arthritis': 'N', 'History of secondary osteoporosis': 'N', 'Alcoholism (> 3 drinks per day)': 'N', 'Medial cortical thickness at 50 mm': 6.599999904632568, 'Lateral cortical thickness at 50 mm': 5.900000095367432, 'Femoral Horizontal Offset': 47.599998474121094, 'Femoral neck angle': 131.89999389648438, 'Femural Head Diameter': 53.900001525878906, 'Femoral Vertical Offset': 76.9000015258789, 'Femoral neck thickness': 37.599998474121094, 'DXA Femoral Neck #1 T-score (g/cm^2)': -2.4000000953674316, 'DXA L1-L4 T-score (g/cm^2)': nan, 'FRAX Score - online calculator major fracture (%)': 28.0, 'FRAX Score - online calculator hip fracture (%)': 6.0})
output_model = create_model("AFF_Prediction_ET_T_scores_output", prediction=1)


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

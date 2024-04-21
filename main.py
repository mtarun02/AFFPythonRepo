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

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from all origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class FormData(BaseModel):
    fieldRace: str
    fieldAgeAtFirst: str
    fieldBisphosphonates: str
    fieldDuration: str
    fieldOtherMedication: str
    fieldWeight: str
    fieldHeight: str
    fieldBMI: str
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
    fieldDXAlumbar: str
    fieldFraxMajor: str
    fieldFraxHip: str

  


model = load_model("AFF_Prediction2")
input_model = create_model
def predict (race,age,bisphosphonate,duration,other,weight,height,bmi,ntf,parent,smoke,steroid,ra,so,alcohol,mct,lct,offset,fma,fmd,vo,fmt,dxa1,dxa2,frax1,frax2):
    
    #fields are inputted as strings. If they are empty strings, set them as NaN so that they can be passed as empty floats and handled by model. 
    if height == "":
        height = "nan"
    if weight == "":
        weight = "nan"
    if bmi == "":
        bmi = "nan"
    if age == "":
        age = "nan"
    if duration == "":
        duration = "nan"
    if mct == "":
        mct = "nan"
    if lct == "":
        lct = "nan"
    if offset == "":
        offset = "nan"
    if fma == "":
        fma = "nan"
    if fmd == "":
        fmd = "nan"
    if vo == "":
        vo = "nan"
    if fmt == "":
        fmt = "nan"
    if dxa1 == "":
        dxa1 = "nan"
    if dxa2 == "":
        dxa2 = "nan"
    if frax1 == "":
        frax1 = "nan"
    if frax2 == "":
        frax2 = "nan"

    data = pd.DataFrame([[race,age,bisphosphonate,duration,other,weight,height,bmi,ntf,parent,smoke,steroid,ra,so,alcohol,mct,lct,offset,fma,fmd,vo,fmt,dxa1,dxa2,frax1,frax2]])
    data.columns = ['Race','Age at fracture (first hip)','Bisphosphonates 1','Duration (yrs)','Other medications 1','Weight (kg)','Height (cm)','BMI (Kg/m^2)','Previous non traumatic fracture','Parent fractured hip (mother or father)','Current smoker','Steroid Use (> 3 months)','Rheumatoid arthritis','History of secondary osteoporosis','Alcoholism (> 3 drinks per day)','Medial cortical thickness','Lateral cortical thickness','Offset','Femoral neck angle','Femural Head Diameter','Vertical Offset','Femoral neck thickness','DXA Femoral Neck #1 BMD (g/cm^2)','DXA L1-L4 BMD (g/cm^2)','FRAX Score - online calculator major fracture (%)','FRAX Score - online calculator hip fracture']
    predictions = predict_model(model,data=data)
    return{'Prediction:': predictions["prediction_label"].iloc[0],"Probability of Prediction":predictions["prediction_score"].iloc[0]}



#@app.post("/submit/")
@app.post("/")

async def submit_form(data: FormData):
    try:
        form_data = data.dict()

        print(form_data.get('fieldRace'))
        print(predict(form_data.get('fieldRace'), form_data.get('fieldAgeAtFirst'), form_data.get('fieldBisphosphonates'), form_data.get('fieldDuration'), form_data.get('fieldOtherMedication'), form_data.get('fieldWeight'), form_data.get('fieldHeight'), form_data.get('fieldBMI'), form_data.get('fieldPreviousNTF'), form_data.get('fieldParent'), form_data.get('fieldSmoker'), form_data.get('fieldSteroid'), form_data.get('fieldRA'), form_data.get('fieldSO'), form_data.get('fieldAlcoholism'), form_data.get('fieldMCT'), form_data.get('fieldLCT'), form_data.get('fieldOffset'), form_data.get('fieldFMA'), form_data.get('fieldFMD'), form_data.get('fieldVO'), form_data.get('fieldFMT'), form_data.get('fieldDXABMD'), form_data.get('fieldDXAlumbar'), form_data.get('fieldFraxMajor'), form_data.get('fieldFraxHip')))
        AFF_prediction = predict(form_data.get('fieldRace'), form_data.get('fieldAgeAtFirst'), form_data.get('fieldBisphosphonates'), form_data.get('fieldDuration'), form_data.get('fieldOtherMedication'), form_data.get('fieldWeight'), form_data.get('fieldHeight'), form_data.get('fieldBMI'), form_data.get('fieldPreviousNTF'), form_data.get('fieldParent'), form_data.get('fieldSmoker'), form_data.get('fieldSteroid'), form_data.get('fieldRA'), form_data.get('fieldSO'), form_data.get('fieldAlcoholism'), form_data.get('fieldMCT'), form_data.get('fieldLCT'), form_data.get('fieldOffset'), form_data.get('fieldFMA'), form_data.get('fieldFMD'), form_data.get('fieldVO'), form_data.get('fieldFMT'), form_data.get('fieldDXABMD'), form_data.get('fieldDXAlumbar'), form_data.get('fieldFraxMajor'), form_data.get('fieldFraxHip'))
        #return JSONResponse(content={"Prediction":AFF_prediction})

        #with open("form_data.json", "w") as json_file:
        #    json.dump(data.dict(), json_file)
        return JSONResponse(content={"message": str(AFF_prediction)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

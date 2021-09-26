from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd

from ml import MODEL_PATH, ENCODER_PATH, ALL_FEATURES, CAT_FEATURES, LABEL, EXAMPLE_CENSUS_DATA_POS, EXAMPLE_CENSUS_DATA_NEG
from ml.data import process_data
from ml.model import inference

import pickle


class CensusData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')

    class Config:
        schema_extra = {
            "example": EXAMPLE_CENSUS_DATA_POS}


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "welcome"}


@app.post("/predict/")
async def predict(data: CensusData):
    assert set(ALL_FEATURES) - \
        set(data.dict(by_alias=True).keys()) == set(
            [LABEL]), "invalid input"

    model = pickle.load(open(MODEL_PATH, 'rb'))
    encoder = pickle.load(open(ENCODER_PATH, 'rb'))

    X, _, _, _ = process_data(X=pd.DataFrame(
        [data.dict(by_alias=True)]), categorical_features=CAT_FEATURES, training=False, encoder=encoder)

    return {"prediction": str(inference(model, X))}

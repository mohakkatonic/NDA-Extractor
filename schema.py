from pydantic import BaseModel

# sample Predict Schema 
# Make sure the key is 'data' and your data can be of any type 
class PredictSchema(BaseModel):
    data: str

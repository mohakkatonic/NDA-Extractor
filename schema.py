from pydantic import BaseModel 
from typing import List, Any

# sample Predict Schema 
# Make sure key is data and your data can be of anytype 
class PredictSchema(BaseModel): 
    data: str
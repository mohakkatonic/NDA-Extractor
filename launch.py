from langchain.llms import OpenAI
from langchain.chains import LLMChain
from pydantic import BaseModel
from typing import List, Any, Dict, Union
import requests
import json
import os
from schema import PredictSchema

def loadmodel(logger):
    """Get the model"""
    openai_model = OpenAI(
        model_name=os.environ.get("MODEL_NAME", "gpt-3.5-turbo-16k"),
        openai_api_key=os.environ.get("API_KEY"),
        temperature=0,
        max_tokens=10000,
        top_p=1.0,
        frequency_penalty=1.0
    )
    logger.info(f"Model fetched")
    return openai_model

def preprocessing(data, logger):
    """ Applies preprocessing techniques to extract specific info from the raw data"""
    logger.info("Task fetched.")

    # Modify the prompt to guide the model in extracting information
    final_prompt =(
        f"""
    You are an NDA (Non-Disclosure Agreement) reader expert, who specializes in reviewing and understanding the intricacies of confidentiality agreements. You possess a deep understanding of legal language and can interpret the terms and conditions outlined in NDAs to ensure clarity and protection of sensitive information for all parties involved. Extract the following information from the given NDA document content:
    Context: {data['data']}.
    Information to extract: Client Name, Contract Start Date, Contract End Date, Special Terms for Renewal.
    Extract based on below definitions:
    Client Name: The legal or business name of the party involved in the NDA.
    Contract Start Date: The specific date when the contractual agreement covered by the NDA begins.
    Contract End Date: The date marking the termination or expiration of the contractual agreement specified in the NDA.
    If the contract end date is not specified directly, in such cases, the duration of confidentiality is often addressed in the "Survival" clause. Calculate the end date from there and give output as a date.
    Special Terms for Renewal: Unique conditions or provisions in the NDA outlining terms for contract renewal.
    Extracted Information:
    """
    )

    logger.info("Created the final Prompt")
    return final_prompt

def predict(final_prompt, openai_model, logger):
    """Predicts the results for the given inputs"""
    logger.info(f"final_prompt: {final_prompt}")
    logger.info("Model prediction started.")
    try:
        response = openai_model(final_prompt)
    except Exception as e:
        logger.info(e)
    logger.info("Prediction Done.")
    return response

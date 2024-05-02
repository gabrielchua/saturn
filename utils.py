"""
Utils.py
"""
import asyncio
from typing import (
    Optional,
    Tuple
    )

import numpy as np
import pandas as pd
import streamlit as st
from openai import AsyncOpenAI

from config import (
    LLM,
    MAX_CONCURRENT_CALLS
    )

client = AsyncOpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def set_page_configurations() -> None:
    """
    Sets page configurations
    """
    st.set_page_config(page_title="SATURN", 
                       page_icon="🪐",
                       layout="wide",)

def set_session_state() -> None:
    """
    Sets session state
    """
    session_state = ["original_df", "tagging_configurations"]
    for state in session_state:
        if state not in st.session_state:
            st.session_state[state] = None
    st.session_state["configured"] = False

def set_custom_css() -> None:
    """
    Applies custom CSS
    """
    st.markdown("""
            <style>
            #MainMenu {visibility: hidden}
            #header {visibility: hidden}
            #footer {visibility: hidden}
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                padding-left: 3rem;
                padding-right: 3rem;
                }
                
            .stApp a:first-child {
                display: none;
            }
            .css-15zrgzn {display: none}
            .css-eczf16 {display: none}
            .css-jn99sy {display: none}
            </style>
            """, unsafe_allow_html=True)

def config_fail_validation(tagging_configurations: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Checks if the configuration df fails validation
    """
    # Check no duplicate rows in the first column
    if tagging_configurations.iloc[:, 0].duplicated().any():
        return True, "Duplicate label categories. Please remove duplicates."
    # Check all descriptions are provided
    if tagging_configurations.iloc[:, 1].isna().any():
        return True, "Some categories are missing descriptions. Please fill in all descriptions."
    return False, None

@st.experimental_fragment
def render_download_page():
    """
    Render download page
    """
    st.download_button("Download Tagged Data", 
                data=st.session_state["tagged_df"].to_csv(index=False), 
                file_name=f"{st.session_state.uploaded_file_name[:-4]}_tagged.csv",
                mime="text/csv")
    
async def start_async_tag_job(dataframe: pd.DataFrame,
                              column_to_classify: str,
                              tagging_configurations: pd.DataFrame) -> pd.DataFrame:
    """
    Tag the dataframe
    """
    # Setup
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)  # Limit to 10 concurrent API calls
    df = dataframe.copy()
    df["original_row_number"] = df.index

    # Convert tagging configuration df to dictionary
    num_categories = tagging_configurations.shape[0]
    tagging_dict = {}
    for i in range(num_categories):
        tagging_dict[i] = {"label": tagging_configurations.iloc[i, 0],
                           "description": tagging_configurations.iloc[i, 1], 
                           "example": tagging_configurations.iloc[i, 2]}
    tagging_dict[9] = {"label": "NOT_TAGGED_UNCLEAR"}

    system_prompt = _generate_system_prompt(tagging_dict)
    logit_bias_dict = _generate_logit_bias_dict(num_categories)

    # Schedule tasks for execution with their original DataFrame index
    scheduled_tasks = [(index, asyncio.create_task(zero_shot_classifier_async(semaphore,
                                                                              row[column_to_classify],
                                                                              system_prompt,
                                                                              logit_bias_dict))) 
                                                                              for index, row in df.iterrows()]

    # Initialize an empty list to store results along with the original DataFrame index
    results = []
    # Correctly await tasks and collect results with index
    for index, task in scheduled_tasks:
        category, confidence = await task
        results.append((index, category, confidence))

    # Sort and apply results as before
    results.sort(key=lambda x: x[0])
    for index, category, confidence in results:
        df.at[index, "Tag"] = category
        df.at[index, "Confidence"] = confidence

    # Map the category to the label
    tag_mapping = {9: "NOT_TAGGED_UNCLEAR"}
    for i in range(num_categories):
        tag_mapping[i] = tagging_dict[i]["label"]
    df["Tag"] = df["Tag"].map(tag_mapping)
    df.loc[df["Tag"] == "NOT_TAGGED_UNCLEAR", "Confidence"] = np.nan
    return df

async def zero_shot_classifier_async(semaphore,
                                     text: str,
                                     system_prompt: str,
                                     logit_bias_dict: dict[str, int]) -> Tuple[int, float]:
    """
    Asynchronous zero-shot classifier using OpenAI's API
    """
    async with semaphore:
        response = await client.chat.completions.create(
            model=LLM,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            max_tokens=1,
            logit_bias=logit_bias_dict,
            temperature=0,
            seed=0,
            logprobs=True
        )
        category = int(response.choices[0].message.content)
        log_prob = response.choices[0].logprobs.content[0].logprob
        confidence = np.exp(log_prob)
        return category, confidence

def _generate_system_prompt(tagging_dict: dict[str, str]) -> str:
    """
    System prompt
    """
    num_categories = len(tagging_dict) - 1
    categories_description = ["Category " + str(i) + ": " + tagging_dict[i]["label"] + " (" + tagging_dict[i]["description"] + ") \n \n " for i in range(num_categories) ]
    examples = ["INPUT: " + tagging_dict[i]["example"] + "\n OUTPUT: " + str(i) + "\n \n" for i in range(num_categories)]
    return f"""Your task is to classify the following text into one of the following categories: \n \n  {" ".join(categories_description)}. \n \n For example: {" ".join(examples)}. If the text does not clearly belong to any of the above categories, return `9`."""

def _generate_logit_bias_dict(num_categories: int) -> dict[str, int]:
    """
    Returns logit bias dictionary
    """
    logit_bias_dict = {}
    for i in range(num_categories):
        logit_bias_dict[str(15+i)] = 100
    logit_bias_dict["24"] = 100
    return logit_bias_dict

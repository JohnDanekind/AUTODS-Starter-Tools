from typing import Annotated, Tuple, List, Dict, Any
from typing_extensions import TypedDict
import pandas as pd
from langchain_core.tools import tool
import os


@tool(response_format='content_and_artifact')
def load_file(file_path: str) -> Tuple[str, Dict]:
    """
     Automatically loads a file based on its extension.

     Parameters:
     ----------
     file_path : str
         The path to the file to load.

     Returns:
     -------
     Tuple[str, Dict]
         A tuple containing a message and a dictionary of the data frame.
     """
    print(f"    * Tool: load_file | {file_path}")
    return f"Returned the following data frame from this file: {file_path}", auto_load_file(file_path).to_dict()


def auto_load_file(filepath: str) -> pd.DataFrame:
    """
    Automatically loads files based on its extension

    parameters
    __________
    file_path: str
        The path to the file

    returns
    _______
    pd.DataFrame
    """
    try:
        file_extension = filepath.split(".")[-1].lower()
        if file_extension == "csv":
            return load_csv(filepath)
        elif file_extension == "json":
            return load_json(filepath)
    except Exception as e:
        return f"Error loading file: {e}"


def load_csv(file_path: str) -> pd.DataFrame:
    """
    Tool: load_csv
    Description: Loads CSV file into pandas dataframe
    Args:
        file_path (str): Path to the csv file

    Returns
        pd.Dataframe
    """
    return pd.read_csv(file_path)


def load_json(file_path: str) -> pd.DataFrame:
    """
    Tool: load_json
    Description: Loads json file into pandas dataframe
    Args:
        file_path (str): Path to the csv file

    Returns
        pd.Dataframe
    """
    return pd.read_json(file_path)

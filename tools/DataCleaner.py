from typing import Annotated, List, Dict, Any
from typing_extensions import TypedDict
import pandas as pd
from langchain_core.tools import tool
import os


@tool
def placeholder_data_cleaning_tool(input: str) -> str:
    """
    This is a placeholder tool that will be used for cleaning datasets later.
    Later it will get rid of NaN values, convert datatypes as specified, and maybe preprocess some stuff
    As of now it just returns a basic string
    :param input:
        String input
    :return:
    """
    return f" I am the placeholder_data_cleaning_tool  and this is the response to the {input}"

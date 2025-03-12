from typing import Annotated, List, Dict, Any
from typing_extensions import TypedDict
import pandas as pd
from langchain_core.tools import tool
import os


@tool()
def basic_eda(filepath: str = None) -> str:
    """
    Perform basic exploratory data analysis on a dataframe

    parameters
    __________
    file_path: str, optional
        Path to the datafile. If none, uses the currently loaded dataframe

    Returns
    _______
    str
        Basic EDA results as a formatted string
    """

    # Access dataframe
    df = None
    if filepath:
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            return f"Error loading file: {str(e)}"
    else:
        return "No dataframe loaded. Please load the dataframe first"

    if df is None:
        return "No DataFrame available for analysis"

    # Start building EDA report
    results = []

    # Get basic dataframe information (shape and list of columnns)
    results.append("## Basic DataFrame Information")
    results.append(f"- Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    results.append(f"- Columns {', '.join(df.columns.tolist())}")

    # Get details on each column of the DataFrame
    results.append(f"\n## Column Details")
    for col in df.columns:
        col_type = str(df[col].dtype)
        results.append(f"\n### {col}, ({col_type})")

        # Statistics based on data type
        if pd.api.types.is_numeric_dtype(df[col]):
            # Numeric column
            stats = df[col].describe()
            results.append(f"- Count: {stats['count']}")
            results.append(f" - Mean: {stats['mean']:.2f}")
            results.append(f" - Std Dev: {stats['std']:.2f}")
            results.append(f" - Min: {stats['min']:.2f}")
            results.append(f" - 25%: {stats['25%']:.2f}")
            results.append(f" - 50%: {stats['50%']:.2f}")
            results.append(f" - 75%: {stats['75%']:.2f}")
            results.append(f" - Max%: {stats['max']:.2f}")
            results.append(f"- Missing Values: {df[col].isna().sum()} ({df[col].isna().mean:.2%})")

        else:
            # Categorical/text column
            results.append(f" - Unique values: {df[col].nunique()}")
            results.append(f"- Missing values: {df[col].isna().sum()} ({df[col].isna().mean():.2%})")

            # Top 5 most common values
            if df[col].nunique() < 50:  # Only for columns with reasonable number of unique values
                top_values = df[col].value_counts().head(5)
                results.append("- Top 5 values:")
                for val, count in top_values.items():
                    results.append(f"  - {val}: {count} ({count / len(df):.2%})")

            # Missing values summary
        results.append("\n## Missing Values Summary")
        missing = df.isnull().sum()
        missing_cols = missing[missing > 0]

        if len(missing_cols) > 0:
            for col, count in missing_cols.items():
                results.append(f"- {col}: {count} missing values ({count / len(df):.2%})")
        else:
            results.append("- No missing values in the dataset")

    return "\n".join(results)


# @tool
# def baby_eda_tool(input: str) -> str:
#     """
#     This is a placeholder tool that will be used for eda later.
#     As of now it just returns a basic string
#     :param input:
#         String input
#     :return:
#     """
#     return f" I am the baby_eda_tool and this is the response to the {input}"

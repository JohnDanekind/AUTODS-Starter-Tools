from typing import Annotated, List, Dict, Any
from typing_extensions import TypedDict
import pandas as pd
from langchain_core.tools import tool


@tool
def simple_df_summary(csv_file_path: str) -> Dict[str, Any]:
    """
    Provide a summary of a DataFrame from a CSV file path.

    Args:
        csv_file_path: Path to the CSV file to analyze

    Returns:
        A dictionary with summary statistics
    """
    # Load the DataFrame inside the function
    df = pd.read_csv(csv_file_path)

    # Create summary
    summary = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_summary": {
            col: {
                "mean": df[col].mean(),
                "median": df[col].median(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max()
            } for col in df.select_dtypes(include=['number']).columns
        }
    }

    return summary




@tool
def analyze_column(csv_file_path: str, column_name: str) -> Dict[str, Any]:
    """
    Analyze a specific column in a CSV file.

    Args:
        csv_file_path: Path to the CSV file
        column_name: Name of the column to analyze

    Returns:
        Statistics for the specified column
    """
    df = pd.read_csv(csv_file_path)

    if column_name not in df.columns:
        return {"error": f"Column '{column_name}' not found. Available columns: {df.columns.tolist()}"}

    col_data = df[column_name]

    if pd.api.types.is_numeric_dtype(col_data):
        return {
            "mean": col_data.mean(),
            "median": col_data.median(),
            "std": col_data.std(),
            "min": col_data.min(),
            "max": col_data.max(),
            "quartiles": [col_data.quantile(0.25), col_data.quantile(0.5), col_data.quantile(0.75)]
        }
    else:
        return {
            "unique_values": col_data.nunique(),
            "most_common": col_data.value_counts().head(5).to_dict(),
            "missing_values": col_data.isnull().sum()
        }


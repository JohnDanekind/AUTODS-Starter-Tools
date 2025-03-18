from typing import Annotated, List, Dict, Optional, Any
from typing_extensions import TypedDict
import pandas as pd
import numpy as np 
from langchain_core.tools import tool
import os
from pydantic import BaseModel

    
    

@tool 
def handle_missing_values(csv_file_path: str, strategy: str = "drop") -> pd.DataFrame:
    """ 
    Handle missing values in the DataFrame based on strategy
    
    args: 
        csv_file_path: Path to the CSV file
        strategy: Strategy to handle missing values - "drop", "mean", "median", "mode", "value"
    
    returns: 
        pd.DataFrame with handled missing values
    """
    df = pd.read_csv(csv_file_path)
    
    if strategy == "drop":
        df.dropna(inplace=True)
    elif strategy == "mean":
        for column in df.select_dtypes(include=[np.number]).columns:
            df[column].fillna(df[column].mean(), inplace=True)
    elif strategy == "median":
        for column in df.select_dtypes(include=[np.number]).columns:
            df[column].fillna(df[column].median(), inplace=True)
    elif strategy == "mode":
        for column in df.columns:
            df[column].fillna(df[column].mode()[0] if not df[column].mode().empty else np.nan, inplace=True)
    elif strategy == "zero":
        for column in df.select_dtypes(include=[np.number]).columns:
            df[column].fillna(0, inplace=True)
    
    print(f"\n {df}")
    return df


@tool
def drop_duplicates(csv_file_path: str, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Drop duplicate rows from the DataFrame
    
    args:
        csv_file_path: Path to the CSV file
        subset: List of column names to consider for identifying duplicates
    
    returns:
        pd.DataFrame without duplicates
    """
    df = pd.read_csv(csv_file_path)
    df.drop_duplicates(subset=subset, inplace=True)
    print(f"\n {df}")
    return df


@tool
def convert_data_types(csv_file_path: str, type_conversions: Dict[str, str]) -> pd.DataFrame:
    """
    Convert data types of columns
    
    args:
        csv_file_path: Path to the CSV file
        type_conversions: Dictionary with column names as keys and desired types as values
                          Supported types: 'int', 'float', 'str', 'bool', 'datetime'
    
    returns:
        pd.DataFrame with converted data types
    """
    df = pd.read_csv(csv_file_path)
    
    for column, dtype in type_conversions.items():
        if column in df.columns:
            if dtype == 'int':
                df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')  # Int64 handles NaN
            elif dtype == 'float':
                df[column] = pd.to_numeric(df[column], errors='coerce')
            elif dtype == 'str':
                df[column] = df[column].astype(str)
            elif dtype == 'bool':
                df[column] = df[column].astype(bool)
            elif dtype == 'datetime':
                df[column] = pd.to_datetime(df[column], errors='coerce')
    
    print(f"\n {df}")
    return df


@tool
def handle_outliers(csv_file_path: str, columns: List[str], method: str = "iqr", threshold: float = 1.5) -> pd.DataFrame:
    """
    Detect and handle outliers in specified columns
    
    args:
        csv_file_path: Path to the CSV file
        columns: List of column names to check for outliers
        method: Method to detect outliers - "iqr", "zscore", "percentile"
        threshold: Threshold for outlier detection (default: 1.5 for IQR, 3 for z-score)
    
    returns:
        pd.DataFrame with outliers handled
    """
    df = pd.read_csv(csv_file_path)
    
    for column in columns:
        if column in df.select_dtypes(include=[np.number]).columns:
            if method == "iqr":
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
            elif method == "zscore":
                mean = df[column].mean()
                std = df[column].std()
                df = df[abs((df[column] - mean) / std) <= threshold]
            elif method == "percentile":
                lower = df[column].quantile(0.01)  # 1st percentile
                upper = df[column].quantile(0.99)  # 99th percentile
                df = df[(df[column] >= lower) & (df[column] <= upper)]
    
    print(f"\n {df}")
    return df

# Make normalizing tool 
# use minmax normalization, standard deviation normalization, and robust normalization
# use the same input and output format as the other tools

@tool 
def normalize_data(csv_file_path: str, columns: List[str], method: str = "minmax") -> pd.DataFrame:
    """
    Normalize numeric columns in the DataFrame

    Args:
        csv_file_path (str): _description_
        columns (List[str]): _description_
        method (str, optional): _description_. Defaults to "minmax".

    Returns:
        pd.DataFrame: with normalized columns
    """
    df = pd.read_csv(csv_file_path)
    
    for column in columns:
        if column in df.select_dtypes(include=[np.number]).columns:
            if method == "minmax":
                min_val = df[column].min()
                max_val = df[column].max()
                df[column] = (df[column] - min_val) / (max_val - min_val)
            elif method == "std":
                mean = df[column].mean()
                std = df[column].std()
                df[column] = (df[column] - mean) / std 
            elif method == "robust":
                q1 = df[column].quantile(0.25)
                q3 = df[column].quantile(0.75)
                IQR = q3 - q1 
                df[column] = (df[column] - q1) / IQR
    
    print(f"\n {df}")
    return df

# Make tool for encoding categorical data
# use one-hot encoding and label encoding
# use the same input and output format as the other tools
@tool 
def encode_categorical_data(csv_file_path: str, columns: List[str], method: str = "label") -> pd.DataFrame:
    """
    Encode categorical columns in the DataFrame

    Args:
        csv_file_path (str): Path to csv file 
        columns (List[str]): Specify the columns to encode
        method (str, optional): Specify the encoding method - "onehot" or "label". Defaults to "onehot".

    Returns:
        pd.DataFrame: with encoded columns
    """
    df = pd.read_csv(csv_file_path)
    
    for column in columns:
        if column in df.columns:
            if method == "onehot":
                dummies = pd.get_dummies(df[column], prefix=column)
                df = pd.concat([df, dummies], axis=1)
                df.drop(column, axis=1, inplace=True)
            elif method == "label":
                df[column] = pd.factorize(df[column])[0]
            
            elif method == "ordinal":
                categories = df[column].unique()
                category_map = {category: i for i, category in enumerate(categories)}
                df[column] = df[column].map(category_map)  
    
    print(f"\n {df}")
    return df


            

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


# AUTODS-Starter-Tools

## Table of Contents 
1. [Introduction](#introduction)
2. [Setup Instructions](#setup)

## Introduction
This is a rough draft of some potential tools that we can integrate into our final project. It is currently a work in progress and some of the tools are buggy but I plan on refining it more and more. I currently have one agent with the following tools:
* DataLoader: Loads csv and json files and coverts them to DataFrames

* DataFrame: This tool will give basic summary information about DataFrames or analyze specifc columns of the DataFrame. I plan on refining it later and making it more like the DataFrame tool from the AI-Data-Science-Team repo. 

* EDA: Does basic EDA on a DataFrame. Shows basic things like DataFrame shape, DataFrame columns, details about each column both numeric and categorical, common values for DataFrame columns, and missing value information for each column in the DataFrame. 

* Visualization: Makes scatter plots, bar charts, and histograms of data using plotly. Doesn't work right now and I plan on changing and fixing it later. 

* mini_tools: These are just dummy tools I made to ensure the agent is calling tools correctly. Can be ignored. 

* I have dummy tools right now for output and DataCleaner. I might make these agents because I am not sure how to make tools to output everything and clean data that consistently work well. 

## Setup Instructions


## Setup


### 1. Clone Repo 
bash 
```
git clone https://github.com/JohnDanekind/AUTODS-Starter-Tools.git
cd AUTODS-Starter-Tools
```

### 2. Set Up a Virtual Environment
You will need python 3.10 or later for some of the later dependencies 
- **macOS / Linux**:
  ```bash
  python -m venv venv
  source venv/bin/activate
  ```
- **Windows**:
  ```bash
  python -m venv venv
  venv\\Scripts\\activate
  ```
### 3. Install Dependencies
Install the required dependencies using `uv pip` (or `pip` if `uv` is not available):
```bash
uv pip install -r requirements.txt
```
*Note: If `uv` is not installed, use `pip install -r requirements.txt`.*

### 4. Create a `.env` File
Create a `.env` file in the root directory and add your OpenAI API key or any langchain complient LLM API key:
```plaintext
OPENAI_API_KEY=your_api_key_here
```

### 5. Run the Application
Execute the main script to start the conversational agent:
```
python app.py
```
---




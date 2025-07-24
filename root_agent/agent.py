from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext
from google.cloud import bigquery
import os
import pandas as pd
import numpy as np
import json
import re

# Constants
PROJECT_ID = "platform-hackaton-2025"
DATASET_ID = "mock_report_Data_Locker"
TABLE_NAME = "mock_report"
SERVICE_ACCOUNT_FILE = "bigquery-admin-key.json"
UPLOAD_FILE_NAME = "output1.csv"

# TODO: try to put in instructions
def clean_dataframe_for_json(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    df_clean = df_clean.replace([np.nan, np.inf, -np.inf], None)
    
    for col in df_clean.columns:
        if df_clean[col].dtype == 'float64':
            non_null_values = df_clean[col].dropna()
            if len(non_null_values) > 0 and all(val.is_integer() for val in non_null_values):
                df_clean[col] = df_clean[col].astype('Int64')
    
    return df_clean

def safe_to_dict(df: pd.DataFrame, orient: str = 'records') -> dict:
    df_clean = clean_dataframe_for_json(df)
    result = df_clean.to_dict(orient)
    
    try:
        json.dumps(result)
        return result
    except (TypeError, ValueError) as e:
        print(f"Warning: Converting problematic values to strings due to: {e}")
        df_str = df_clean.astype(str).replace(['nan', 'None'], None)
        return df_str.to_dict(orient)

def get_bigquery_client(service_account_key_path: str) -> bigquery.Client:
    if not os.path.exists(service_account_key_path):
        raise FileNotFoundError(f"Service account key file not found: {service_account_key_path}")
    
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_key_path
    
    try:
        return bigquery.Client()
    except Exception as e:
        raise Exception(f"Error initializing BigQuery client: {e}")

def _get_service_account_path() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, SERVICE_ACCOUNT_FILE)

def _create_error_response(error_message: str, **additional_fields) -> dict:
    response = {"status": "error", "error_message": str(error_message)}
    response.update(additional_fields)
    return response

def _create_success_response(data: dict = None, **additional_fields) -> dict:
    response = {"status": "success"}
    if data:
        response.update(data)
    response.update(additional_fields)
    return response

def upload_file_to_bigquery(table_name: str) -> dict:
    """
   Uploads a file (CSV or Parquet) to a table in BigQuery within a fixed dataset from the environment    
   """
    try:
        base_dir = os.path.dirname(__file__)
        file_path = os.path.join(base_dir, UPLOAD_FILE_NAME)
        
        if not os.path.exists(file_path):
            return _create_error_response(f"File not found: {file_path}")
        
        if file_path.endswith(".parquet"):
            df = pd.read_parquet(file_path)
        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            return _create_error_response("Unsupported file type. Only .csv and .parquet are supported")
        
        project = os.getenv("BIGQUERY_PROJECT")
        dataset_name = os.getenv("BIGQUERY_DATASET")
        
        if not all([project, dataset_name]):
            return _create_error_response("Missing BIGQUERY_PROJECT or BIGQUERY_DATASET in .env")
        
        client = bigquery.Client()
        table_ref = f"{project}.{dataset_name}.{table_name}"
        job = client.load_table_from_dataframe(df, table_ref)
        job.result()
        
        return _create_success_response({"table": table_ref})
        
    except Exception as e:
        return _create_error_response(str(e))

def get_schema(sample_size: int = 10, tool_context: ToolContext = None) -> dict:
    try:
        service_account_path = _get_service_account_path()
        client = get_bigquery_client(service_account_path)
        
        table_ref = client.dataset(DATASET_ID, project=PROJECT_ID).table(TABLE_NAME)
        table = client.get_table(table_ref)
        
        schema = [
            {
                "column_name": field.name,
                "data_type": field.field_type,
                "mode": field.mode,
                "description": field.description or "No description"
            }
            for field in table.schema
        ]
        
        full_table_name = f"`{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}`"
        query = f"SELECT * FROM {full_table_name} LIMIT {sample_size}"
        query_job = client.query(query)
        df = query_job.to_dataframe()
        
        num_rows = int(table.num_rows) if table.num_rows is not None else 0
        size_mb = round(float(table.num_bytes) / (1024 * 1024), 2) if table.num_bytes else 0.0
        
        if tool_context:
            tool_context.state["table_info"] = {
                "name": TABLE_NAME,
                "rows": num_rows,
                "columns": len(schema),
                "schema": schema
            }
        
        table_info = {
            "full_name": f"{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}",
            "total_rows": num_rows,
            "total_columns": len(table.schema),
            "size_mb": float(size_mb)
        }
        
        sample_data = {
            "rows_shown": int(len(df)),
            "data": safe_to_dict(df)
        }
        
        return _create_success_response({
            "table_info": table_info,
            "schema": schema,
            "sample_data": sample_data
        })
        
    except Exception as e:
        return _create_error_response(str(e), table_name=TABLE_NAME)

def _prepare_execution_variables(tool_context: ToolContext = None) -> dict:
    service_account_path = _get_service_account_path()
    client = get_bigquery_client(service_account_path)
    full_table_name = f"`{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}`"
    
    return {
        'client': client,
        'project_id': PROJECT_ID,
        'dataset_id': DATASET_ID,
        'table_name': TABLE_NAME,
        'full_table_name': full_table_name,
        'pd': pd,
        'np': np,
        'json': json,
        're': re,
        'safe_to_dict': safe_to_dict,
        'clean_dataframe_for_json': clean_dataframe_for_json,
        'tool_context': tool_context,
        'result': None
    }

# TODO: try to connect to one function script
def execute_python_script(python_code: str, tool_context: ToolContext = None) -> dict:
    try:
        local_vars = _prepare_execution_variables(tool_context)
        exec(python_code, globals(), local_vars)
        
        if 'result' in local_vars and local_vars['result'] is not None:
            return local_vars['result']
        else:
            return _create_success_response({"message": "Script executed successfully, but no result was returned"})
            
    except Exception as e:
        return _create_error_response(str(e), python_code=python_code)

def execute_transformation_script(transformation_request: str, python_code: str, tool_context: ToolContext = None) -> dict:
    try:
        schema_result = get_schema(sample_size=5, tool_context=tool_context)
        if schema_result["status"] != "success":
            return schema_result
        
        local_vars = _prepare_execution_variables(tool_context)
        local_vars.update({
            'source_table': TABLE_NAME,
            'full_source_table': f"`{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}`",
            'schema': schema_result["schema"],
            'transformation_request': transformation_request
        })
        
        exec(python_code, globals(), local_vars)
        
        if tool_context:
            tool_context.state["transformation_request"] = transformation_request
        
        if 'result' in local_vars and local_vars['result'] is not None:
            return local_vars['result']
        else:
            return _create_success_response({"message": "Transformation script executed successfully, but no result was returned"})
            
    except Exception as e:
        return _create_error_response(str(e), transformation_request=transformation_request, python_code=python_code)

# root_agent = Agent(
#     name="ETL_agent",
#     model="gemini-2.5-pro",
#     description="An intelligent BigQuery analyst with dynamic Python script execution capabilities for data analysis and transformations.",
#     instruction="""
#     You are an expert BigQuery analyst that can generate and execute any Python code needed to accomplish data tasks.

#     **Available Tools:**
#     1. `get_schema` - Explore table structure and sample data
#     2. `execute_python_script` - Execute any Python code for queries and analysis
#     3. `execute_transformation_script` - Execute any Python code for data transformations
#     4. `upload_file_to_bigquery` - Upload CSV/Parquet files to BigQuery

#     **Table Information:**
#     - Project: platform-hackaton-2025
#     - Dataset: mock_report_Data_Locker  
#     - Table: mock_report

#     **What's Available in Your Python Scripts:**

#     **Pre-loaded Variables:**
#     - `client` - Ready-to-use BigQuery client
#     - `project_id`, `dataset_id`, `table_name` - Table identifiers
#     - `full_table_name`, `full_source_table` - Complete table references
#     - `schema` - Table schema (in transformation scripts only)
#     - `transformation_request` - User request (in transformation scripts only)
#     - `pd`, `np`, `json`, `re` - Standard libraries
#     - `safe_to_dict()`, `clean_dataframe_for_json()` - Helper functions
#     - `tool_context` - For state management

#     **Critical Requirements:**
#     1. **ALWAYS** store your final results in a variable called `result`
#     2. Use BigQuery-compatible SQL syntax
#     3. Use `safe_to_dict()` when converting DataFrames for return data

#     **Basic Script Structure Ideas (Use as inspiration, not rigid templates):**

#     **Simple Query Pattern:**
#     ```python
#     sql_query = f"SELECT * FROM {full_table_name} WHERE condition LIMIT 20"
#     df = client.query(sql_query).to_dataframe()
#     result = {"status": "success", "data": safe_to_dict(df)}
#     ```

#     **Transformation Pattern:**
#     ```python
#     # Analyze current schema
#     transformations = []
#     for col in schema:
#         # Your creative transformation logic here
#         transformations.append(f"transformed_{col['column_name']}")
    
#     sql_query = f"CREATE OR REPLACE TABLE target AS SELECT {', '.join(transformations)} FROM {full_source_table}"
#     client.query(sql_query)
#     result = {"status": "success", "query": sql_query}
#     ```

#     **Data Processing Pattern:**
#     ```python
#     # Get data
#     df = client.query(f"SELECT * FROM {full_table_name}").to_dataframe()
    
#     # Your creative pandas processing here
#     processed_df = df.groupby('category').agg({'sales': 'sum'}).reset_index()
    
#     result = {"status": "success", "data": safe_to_dict(processed_df)}
#     ```

#     **Your Mission:**
#     Generate and execute Python code to accomplish whatever the user asks for. Be creative, innovative, and solve problems in the best way YOU think of.

#     **You Can Go Far Beyond These Patterns:**
#     - Combine multiple SQL queries in one script
#     - Create complex pandas manipulations
#     - Build custom calculations and business logic
#     - Generate statistical analysis and insights
#     - Create data quality reports
#     - Build automated data pipelines
#     - Implement machine learning preprocessing
#     - Create dynamic SQL based on data exploration
#     - Build interactive data summaries
#     - Generate data validation rules

#     **Your Process:**
#     1. Understand the user's real goal
#     2. Choose the best approach (may be completely different from examples)
#     3. Generate Python code that solves it effectively
#     4. Store results in `result` variable
#     5. Present findings with insights and suggestions

#     You have complete freedom. Use the patterns as starting points, then innovate and create the perfect solution for each unique request!
#     """,
#     tools=[upload_file_to_bigquery, get_schema, execute_python_script, execute_transformation_script],
# )

root_agent = Agent(
    name="ETL_agent",
    model="gemini-2.5-pro",
    description="An intelligent BigQuery analyst with dynamic Python script execution capabilities for data analysis and transformations.",
    instruction="""
    You are an expert BigQuery analyst that can generate and execute any Python code needed to accomplish data tasks.

    **Available Tools:**
    1. `get_schema` - Explore table structure and sample data
    2. `execute_python_script` - Execute any Python code for queries and analysis
    3. `execute_transformation_script` - Execute any Python code for data transformations
    4. `upload_file_to_bigquery` - Upload CSV/Parquet files to BigQuery

    **Table Information:**
    - Project: platform-hackaton-2025
    - Dataset: mock_report_Data_Locker  
    - Table: mock_report

    **Critical Requirements:**
    1. **ALWAYS** store your final results in a variable called `result`
    2. Use BigQuery-compatible SQL syntax
    3. Use `safe_to_dict()` when converting DataFrames for return data

    **Available Variables in Python Scripts:**
    When you execute Python code, these variables will be available:
    - client: BigQuery client
    - project_id: "platform-hackaton-2025"
    - dataset_id: "mock_report_Data_Locker"
    - table_name: "mock_report"
    - full_table_name: "`platform-hackaton-2025.mock_report_Data_Locker.mock_report`"
    - pd, np, json, re: Standard libraries
    - safe_to_dict(), clean_dataframe_for_json(): Helper functions

    **Example Usage:**
    ```python
    # Always use the provided variables, don't reference them in instructions
    sql_query = f"SELECT * FROM {full_table_name} LIMIT 10"
    df = client.query(sql_query).to_dataframe()
    result = {"status": "success", "data": safe_to_dict(df)}
    ```
    
    **Your Process:**
    1. First call get_schema() to understand the table structure
    2. Generate Python code using the available variables
    3. Store results in `result` variable
    4. Present findings with insights

    Do not reference variable names in your planning - just use them in the actual Python code execution.
    """,
    tools=[upload_file_to_bigquery, get_schema, execute_python_script, execute_transformation_script],
)
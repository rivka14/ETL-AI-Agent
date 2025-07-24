from google.adk.tools.tool_context import ToolContext
from google.cloud import bigquery
import os
import pandas as pd
import numpy as np
import json
import re
from typing import Optional

#try to put in instructions
def clean_dataframe_for_json(df):
    df_clean = df.copy()
    df_clean = df_clean.replace([np.nan, np.inf, -np.inf], None)
    for col in df_clean.columns:
        if df_clean[col].dtype == 'float64':
            non_null_values = df_clean[col].dropna()
            if len(non_null_values) > 0 and all(val.is_integer() for val in non_null_values):
                df_clean[col] = df_clean[col].astype('Int64')
    return df_clean

def safe_to_dict(df, orient='records'):
    df_clean = clean_dataframe_for_json(df)
    result = df_clean.to_dict(orient)
    try:
        json.dumps(result)
        return result
    except (TypeError, ValueError) as e:
        print(f"Warning: Converting problematic values to strings due to: {e}")
        df_str = df_clean.astype(str).replace(['nan', 'None'], None)
        return df_str.to_dict(orient)

def get_bigquery_client(service_account_key_path: str):
    if not os.path.exists(service_account_key_path):
        raise FileNotFoundError(f"Service account key file not found: {service_account_key_path}")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_key_path
    try:
        client = bigquery.Client()
        return client
    except Exception as e:
        raise Exception(f"Error initializing BigQuery client: {e}")

def upload_file_to_bigquery( table_name: str) -> dict:
    """
   Uploads a file (CSV or Parquet) to a table in BigQuery within a fixed dataset from the environment    
   """
    try:
        base_dir = os.path.dirname(__file__)
        file_path = os.path.join(base_dir, "output1.csv")
        if not os.path.exists(file_path):
            return {"status": "error", "error": f"File not found: {file_path}"}
        if file_path.endswith(".parquet"):
            df = pd.read_parquet(file_path)
        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            return {"status": "error", "error": "Unsupported file type. Only .csv and .parquet are supported"}
        client = bigquery.Client()
        project = os.getenv("BIGQUERY_PROJECT")
        dataset_name = os.getenv("BIGQUERY_DATASET")

        if not all([project, dataset_name]):
            return {"status": "error", "error": "Missing BIGQUERY_PROJECT or BIGQUERY_DATASET in .env"}
    
        table_ref = f"{project}.{dataset_name}.{table_name}"
        job = client.load_table_from_dataframe(df, table_ref)
        job.result()
        return {"status": "success", "table": table_ref}

    except Exception as e:
        return {"status": "error", "error": str(e)}

def get_schema(sample_size: int = 10, tool_context: ToolContext = None) -> dict:
    project_id = "platform-hackaton-2025"
    dataset_id = "mock_report_Data_Locker"
    table_name = "mock_report"

    try:       
        service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        client = get_bigquery_client(service_account_path)
        table_ref = client.dataset(dataset_id, project=project_id).table(table_name)
        table = client.get_table(table_ref)
        schema = [{"column_name": field.name, "data_type": field.field_type, "mode": field.mode, "description": field.description or "No description"} for field in table.schema]
        query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_name}` LIMIT {sample_size}"
        query_job = client.query(query)
        df = query_job.to_dataframe()
        num_rows = int(table.num_rows) if table.num_rows is not None else 0
        size_mb = round(float(table.num_bytes) / (1024 * 1024), 2) if table.num_bytes else 0.0
        if tool_context:
            tool_context.state["table_info"] = {"name": table_name, "rows": num_rows, "columns": len(schema), "schema": schema}
        return {"status": "success", "table_info": {"full_name": f"{project_id}.{dataset_id}.{table_name}", "total_rows": num_rows, "total_columns": len(table.schema), "size_mb": float(size_mb)}, "schema": schema, "sample_data": {"rows_shown": int(len(df)), "data": safe_to_dict(df)}}
    except Exception as e:
        return {"status": "error", "error_message": str(e), "table_name": table_name}

def execute_script(python_code: str, tool_context: ToolContext = None, transformation_request: Optional[str] = None, use_schema: bool = False) -> dict:
    try:
        project_id = "platform-hackaton-2025"
        dataset_id = "mock_report_Data_Locker"
        table_name = "mock_report"
        full_table_name = f"`{project_id}.{dataset_id}.{table_name}`"       
        service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        client = get_bigquery_client(service_account_path)
        
        schema = None
        if use_schema:
            schema_result = get_schema(sample_size=5, tool_context=tool_context)
            if schema_result["status"] != "success":
                return schema_result
            schema = schema_result["schema"]
        
        local_vars = {
            'client': client,
            'project_id': project_id,
            'dataset_id': dataset_id,
            'table_name': table_name,
            'full_table_name': full_table_name,
            'source_table': table_name,
            'full_source_table': full_table_name,
            'transformation_request': transformation_request,
            'schema': schema,
            'pd': pd,
            'np': np,
            'json': json,
            're': re,
            'safe_to_dict': safe_to_dict,
            'clean_dataframe_for_json': clean_dataframe_for_json,
            'tool_context': tool_context,
            'result': None,
        }

        exec(python_code, globals(), local_vars)

        if transformation_request and tool_context:
            tool_context.state["transformation_request"] = transformation_request

        if 'result' in local_vars and local_vars['result'] is not None:
            return local_vars['result']
        else:
            return {"status": "success", "message": "Script executed successfully, but no result was returned"}

    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "python_code": python_code,
            "transformation_request": transformation_request
        }
import os
from typing import Dict, Optional, Union
import pandas as pd
from google.cloud import bigquery


def upload_file_to_bigquery(
    table_name: str, 
    file_path: Optional[str] = None
) -> Dict[str, Union[str, Dict]]:
    """
    Upload a CSV or Parquet file to a BigQuery table.
    
    This function loads data from a file into a specified BigQuery table.
    The dataset is configured through environment variables.
    
    Args:
        table_name: Name of the target BigQuery table.
        file_path: Path to the file to upload. If None, defaults to 'output1.csv' 
                  in the same directory as this script.
    
    Returns:
        dict: Operation result with the following structure:
            - On success: {"status": "success", "table": "<full_table_reference>", 
                          "rows_uploaded": <number_of_rows>}
            - On error: {"status": "error", "error": "<error_message>"}
    
    Raises:
        No exceptions are raised; all errors are returned in the response dict.
    
    Environment Variables:
        BIGQUERY_PROJECT: GCP project ID containing the dataset
        BIGQUERY_DATASET: BigQuery dataset name
    
    Example:
        >>> result = upload_file_to_bigquery("sales_data", "data/sales_2024.csv")
        >>> if result["status"] == "success":
        ...     print(f"Uploaded to {result['table']}")
    """
    try:
        if file_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(base_dir, "file1.csv")
        
        if not os.path.exists(file_path):
            return {
                "status": "error", 
                "error": f"File not found: {file_path}"
            }
        
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension not in [".csv", ".parquet"]:
            return {
                "status": "error", 
                "error": f"Unsupported file type '{file_extension}'. Only .csv and .parquet are supported"
            }
        
        try:
            if file_extension == ".parquet":
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path)
        except Exception as e:
            return {
                "status": "error", 
                "error": f"Failed to read file: {str(e)}"
            }
        
        if df.empty:
            return {
                "status": "error", 
                "error": "File contains no data"
            }
        
        try:
            client = bigquery.Client()
        except Exception as e:
            return {
                "status": "error", 
                "error": f"Failed to initialize BigQuery client: {str(e)}"
            }
        
        project = os.getenv("BIGQUERY_PROJECT")
        dataset_name = os.getenv("BIGQUERY_DATASET")
        
        if not project:
            return {
                "status": "error", 
                "error": "Missing BIGQUERY_PROJECT in environment variables"
            }
        
        if not dataset_name:
            return {
                "status": "error", 
                "error": "Missing BIGQUERY_DATASET in environment variables"
            }
        
        if not table_name or not table_name.strip():
            return {
                "status": "error", 
                "error": "Table name cannot be empty"
            }
        
        table_name = table_name.strip()
        if not all(c.isalnum() or c in ['_', '-'] for c in table_name):
            return {
                "status": "error", 
                "error": "Table name can only contain alphanumeric characters, underscores, and hyphens"
            }
        
        table_ref = f"{project}.{dataset_name}.{table_name}"
        
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            autodetect=True
        )
        
        try:
            job = client.load_table_from_dataframe(
                df, 
                table_ref, 
                job_config=job_config
            )
            job.result()
            
            return {
                "status": "success", 
                "table": table_ref,
                "rows_uploaded": len(df),
                "columns": list(df.columns)
            }
            
        except Exception as e:
            return {
                "status": "error", 
                "error": f"Failed to upload to BigQuery: {str(e)}"
            }
        
    except Exception as e:
        return {
            "status": "error", 
            "error": f"Unexpected error: {str(e)}"
        }
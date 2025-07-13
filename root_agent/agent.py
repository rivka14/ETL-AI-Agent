from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext
import pandas as pd
import pyarrow.parquet as pq
from typing import Dict, List, Any, Optional
import logging
import os
import re


def process_parquet_file(
    file_path: str,
    transformations: str = "convert_numbers_to_int",
    tool_context: ToolContext = None
) -> dict:
    """
    Process a parquet file with specified transformations.
    
    Args:
        file_path: Path to the parquet file
        transformations: Transformation instructions (e.g., "convert_numbers_to_int", "filter_data", etc.)
        tool_context: Tool context for state management
    
    Returns:
        dict: Processing results and metadata
    """
    print(f"--- Tool: process_parquet_file called ---")
    print(f"File: {file_path}")
    print(f"Transformations: {transformations}")
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return {
                "status": "error",
                "error_message": f"File not found: {file_path}",
                "file_path": file_path,
                "suggestions": ["Check the file path", "Ensure the file exists", "Use absolute path"]
            }
        
        # Load parquet file
        df = pd.read_parquet(file_path)
        original_shape = df.shape
        original_dtypes = df.dtypes.to_dict()
        
        print(f"Loaded {len(df)} rows from {file_path}")
        
        # Parse transformations
        transformation_config = parse_transformation_instructions(transformations)
        
        # Apply transformations
        processed_df = apply_data_transformations(df, transformation_config)
        
        # Calculate transformation summary
        transformation_summary = calculate_transformation_summary(df, processed_df, transformation_config)
        
        # Store in context
        if tool_context:
            tool_context.state["last_file_processed"] = file_path
            tool_context.state["last_transformation"] = transformations
            tool_context.state["processed_rows"] = len(processed_df)
        
        return {
            "status": "success",
            "operation": "process_parquet_file",
            "file_path": file_path,
            "original_shape": {
                "rows": original_shape[0],
                "columns": original_shape[1]
            },
            "processed_shape": {
                "rows": processed_df.shape[0],
                "columns": processed_df.shape[1]
            },
            "original_dtypes": {k: str(v) for k, v in original_dtypes.items()},
            "processed_dtypes": {k: str(v) for k, v in processed_df.dtypes.to_dict().items()},
            "column_names": list(processed_df.columns),
            "sample_data": processed_df.head(5).to_dict('records'),
            "transformation_applied": transformation_config,
            "transformation_summary": transformation_summary,
            "data_quality": {
                "null_counts": processed_df.isnull().sum().to_dict(),
                "duplicate_rows": int(processed_df.duplicated().sum()),
                "memory_usage_mb": round(processed_df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
            },
            "message": f"Successfully processed {len(processed_df)} rows with {len(transformation_config)} transformations"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "operation": "process_parquet_file",
            "error_message": str(e),
            "file_path": file_path,
            "error_type": type(e).__name__,
            "suggestions": [
                "Check if the file is a valid parquet format",
                "Ensure sufficient memory for processing",
                "Verify file permissions"
            ]
        }


def analyze_parquet_structure(
    file_path: str,
    tool_context: ToolContext = None
) -> dict:
    """
    Analyze the structure and content of a parquet file without transformations.
    
    Args:
        file_path: Path to the parquet file
        tool_context: Tool context for state management
    
    Returns:
        dict: File analysis results
    """
    print(f"--- Tool: analyze_parquet_structure called ---")
    print(f"File: {file_path}")
    
    try:
        if not os.path.exists(file_path):
            return {
                "status": "error",
                "operation": "analyze_parquet_structure",
                "error_message": f"File not found: {file_path}",
                "file_path": file_path,
                "suggestions": ["Verify file path", "Check file exists", "Use absolute path"]
            }
        
        # Load parquet file
        df = pd.read_parquet(file_path)
        
        # Analyze data types
        numeric_columns = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        string_columns = df.select_dtypes(include=['object', 'string']).columns.tolist()
        datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
        boolean_columns = df.select_dtypes(include=['bool']).columns.tolist()
        
        # Check which numeric columns can be converted to int
        convertible_to_int = []
        conversion_details = {}
        
        for col in numeric_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                try:
                    non_null_values = df[col].dropna()
                    if len(non_null_values) > 0:
                        is_integer = non_null_values.apply(lambda x: float(x).is_integer()).all()
                        if is_integer:
                            convertible_to_int.append(col)
                            conversion_details[col] = {
                                "can_convert": True,
                                "reason": "All values are whole numbers"
                            }
                        else:
                            conversion_details[col] = {
                                "can_convert": False,
                                "reason": "Contains decimal values"
                            }
                    else:
                        convertible_to_int.append(col)
                        conversion_details[col] = {
                            "can_convert": True,
                            "reason": "All values are null"
                        }
                except Exception as e:
                    conversion_details[col] = {
                        "can_convert": False,
                        "reason": f"Error checking: {str(e)}"
                    }
        
        # Get basic statistics
        basic_stats = {
            "total_rows": int(len(df)),
            "total_columns": int(len(df.columns)),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            "file_size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2)
        }
        
        # Get null analysis
        null_analysis = {}
        for col in df.columns:
            null_count = int(df[col].isnull().sum())
            null_analysis[col] = {
                "null_count": null_count,
                "null_percentage": round((null_count / len(df)) * 100, 2),
                "has_nulls": null_count > 0
            }
        
        # Store in context
        if tool_context:
            tool_context.state["last_analyzed_file"] = file_path
            tool_context.state["file_columns"] = list(df.columns)
            tool_context.state["convertible_columns"] = convertible_to_int
        
        return {
            "status": "success",
            "operation": "analyze_parquet_structure",
            "file_path": file_path,
            "basic_stats": basic_stats,
            "column_analysis": {
                "total_columns": len(df.columns),
                "numeric_columns": numeric_columns,
                "string_columns": string_columns,
                "datetime_columns": datetime_columns,
                "boolean_columns": boolean_columns,
                "convertible_to_int": convertible_to_int
            },
            "data_types": {k: str(v) for k, v in df.dtypes.to_dict().items()},
            "conversion_analysis": conversion_details,
            "null_analysis": null_analysis,
            "sample_data": df.head(3).to_dict('records'),
            "data_quality": {
                "has_nulls": df.isnull().any().any(),
                "has_duplicates": df.duplicated().any(),
                "complete_rows": int(len(df.dropna())),
                "completeness_percentage": round((len(df.dropna()) / len(df)) * 100, 2)
            },
            "recommendations": generate_recommendations(df, convertible_to_int)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "operation": "analyze_parquet_structure",
            "error_message": str(e),
            "file_path": file_path,
            "error_type": type(e).__name__,
            "suggestions": [
                "Verify the file is in parquet format",
                "Check file is not corrupted",
                "Ensure sufficient permissions to read file"
            ]
        }


def create_sample_parquet(
    output_path: str = "sample_data.parquet",
    data_type: str = "appsflyer",
    tool_context: ToolContext = None
) -> dict:
    """
    Create a sample parquet file for testing ETL operations.
    
    Args:
        output_path: Path where to save the sample file
        data_type: Type of sample data to create ("appsflyer", "general", "mixed")
        tool_context: Tool context for state management
    
    Returns:
        dict: File creation results
    """
    print(f"--- Tool: create_sample_parquet called ---")
    print(f"Output path: {output_path}")
    print(f"Data type: {data_type}")
    
    try:
        # Create sample data based on type
        if data_type.lower() == "appsflyer":
            sample_data = create_appsflyer_sample_data()
        elif data_type.lower() == "mixed":
            sample_data = create_mixed_sample_data()
        else:
            sample_data = create_general_sample_data()
        
        df = pd.DataFrame(sample_data)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save to parquet
        df.to_parquet(output_path, index=False)
        
        # Analyze what was created
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        convertible_to_int = []
        
        for col in numeric_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                try:
                    if df[col].apply(lambda x: float(x).is_integer()).all():
                        convertible_to_int.append(col)
                except:
                    pass
        
        # Store in context
        if tool_context:
            tool_context.state["sample_file_created"] = output_path
            tool_context.state["sample_data_shape"] = df.shape
            tool_context.state["sample_columns"] = list(df.columns)
        
        return {
            "status": "success",
            "operation": "create_sample_parquet",
            "file_created": output_path,
            "file_size_mb": round(os.path.getsize(output_path) / (1024 * 1024), 4),
            "data_type": data_type,
            "data_shape": {
                "rows": df.shape[0],
                "columns": df.shape[1]
            },
            "columns": list(df.columns),
            "data_types": {k: str(v) for k, v in df.dtypes.to_dict().items()},
            "numeric_columns": numeric_columns,
            "convertible_to_int": convertible_to_int,
            "sample_preview": df.head(3).to_dict('records'),
            "file_info": {
                "exists": os.path.exists(output_path),
                "readable": os.access(output_path, os.R_OK),
                "absolute_path": os.path.abspath(output_path)
            },
            "message": f"Sample parquet file created successfully at {output_path}",
            "next_steps": [
                f"Use analyze_parquet_structure('{output_path}') to examine the file",
                f"Use process_parquet_file('{output_path}', 'convert_numbers_to_int') to test transformations"
            ]
        }
        
    except Exception as e:
        return {
            "status": "error",
            "operation": "create_sample_parquet",
            "error_message": str(e),
            "output_path": output_path,
            "error_type": type(e).__name__,
            "suggestions": [
                "Check if output directory is writable",
                "Ensure sufficient disk space",
                "Verify pandas and pyarrow are installed"
            ]
        }


# Helper functions that return data (not dicts, as they're internal)
def parse_transformation_instructions(instructions: str) -> dict:
    """Parse transformation instructions from user input"""
    instructions_lower = instructions.lower()
    
    config = {
        "convert_numbers_to_int": False,
        "filter_conditions": [],
        "select_columns": None,
        "rename_columns": {},
        "remove_nulls": False,
        "deduplicate": False
    }
    
    # Check for integer conversion
    if any(phrase in instructions_lower for phrase in [
        "convert_numbers_to_int", "numbers to int", "make integers", 
        "convert to int", "all numbers as int", "integer conversion",
        "numbers as type int"
    ]):
        config["convert_numbers_to_int"] = True
    
    # Check for null removal
    if "remove nulls" in instructions_lower or "drop nulls" in instructions_lower:
        config["remove_nulls"] = True
    
    # Check for deduplication
    if "remove duplicates" in instructions_lower or "deduplicate" in instructions_lower:
        config["deduplicate"] = True
    
    return config


def apply_data_transformations(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply transformations to the dataframe"""
    result_df = df.copy()
    
    # Convert numbers to integers
    if config.get("convert_numbers_to_int", False):
        result_df = convert_numeric_to_int(result_df)
    
    # Remove nulls
    if config.get("remove_nulls", False):
        result_df = result_df.dropna()
    
    # Remove duplicates
    if config.get("deduplicate", False):
        result_df = result_df.drop_duplicates()
    
    # Apply filters (if any)
    for filter_condition in config.get("filter_conditions", []):
        result_df = result_df.query(filter_condition)
    
    # Select specific columns (if specified)
    if config.get("select_columns"):
        columns = config["select_columns"]
        if isinstance(columns, list):
            result_df = result_df[columns]
    
    # Rename columns (if specified)
    if config.get("rename_columns"):
        result_df = result_df.rename(columns=config["rename_columns"])
    
    return result_df


def convert_numeric_to_int(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric columns to integers where possible"""
    result_df = df.copy()
    
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            try:
                # Check if all non-null values are whole numbers
                non_null_values = df[column].dropna()
                if len(non_null_values) > 0:
                    # Check if all values are integers (no decimal part)
                    if non_null_values.apply(lambda x: float(x).is_integer()).all():
                        result_df[column] = df[column].astype('Int64')  # Nullable integer
                        print(f"Converted {column} to integer")
                    else:
                        print(f"Skipped {column} - contains decimal values")
                else:
                    # All values are null, convert to nullable int
                    result_df[column] = df[column].astype('Int64')
                    print(f"Converted {column} (all nulls) to integer")
            except (ValueError, TypeError, OverflowError) as e:
                print(f"Could not convert {column} to integer: {e}")
    
    return result_df


def calculate_transformation_summary(original_df: pd.DataFrame, processed_df: pd.DataFrame, config: dict) -> dict:
    """Calculate summary of transformations applied"""
    return {
        "transformations_applied": len([k for k, v in config.items() if v]),
        "rows_changed": len(original_df) != len(processed_df),
        "columns_changed": len(original_df.columns) != len(processed_df.columns),
        "data_types_changed": not original_df.dtypes.equals(processed_df.dtypes),
        "row_difference": len(processed_df) - len(original_df),
        "column_difference": len(processed_df.columns) - len(original_df.columns),
        "config_applied": config
    }


def generate_recommendations(df: pd.DataFrame, convertible_to_int: list) -> list:
    """Generate recommendations based on data analysis"""
    recommendations = []
    
    if convertible_to_int:
        recommendations.append(f"Consider converting {len(convertible_to_int)} numeric columns to integers: {convertible_to_int}")
    
    if df.isnull().any().any():
        recommendations.append("Data contains null values - consider handling them appropriately")
    
    if df.duplicated().any():
        recommendations.append("Data contains duplicate rows - consider deduplication")
    
    if len(df) > 10000:
        recommendations.append("Large dataset detected - consider processing in chunks for better performance")
    
    return recommendations


def create_appsflyer_sample_data() -> dict:
    """Create AppsFlyer-like sample data"""
    return {
        'app_id': ['com.example.app1', 'com.example.app2', 'com.example.app3', 'com.example.app4', 'com.example.app5'],
        'install_time': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']),
        'installs': [100.0, 200.0, 150.0, 300.0, 250.0],
        'clicks': [1000.0, 2000.0, 1500.0, 3000.0, 2500.0],
        'impressions': [10000.0, 20000.0, 15000.0, 30000.0, 25000.0],
        'revenue': [50.5, 75.25, 60.0, 120.0, 95.5],  # Contains decimals
        'conversions': [5.0, 10.0, 8.0, 15.0, 12.0],  # Can be converted to int
        'media_source': ['facebook', 'google', 'apple', 'twitter', 'tiktok'],
        'campaign_name': ['Campaign A', 'Campaign B', 'Campaign C', 'Campaign D', 'Campaign E'],
        'cost': [25.75, 40.50, 30.25, 65.00, 50.75]
    }


def create_mixed_sample_data() -> dict:
    """Create mixed type sample data for testing"""
    return {
        'id': [1.0, 2.0, 3.0, 4.0, 5.0],  # Can convert to int
        'score': [85.5, 92.0, 78.5, 96.0, 88.5],  # Mixed - some can convert
        'count': [10.0, 20.0, 15.0, 30.0, 25.0],  # Can convert to int
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'active': [True, False, True, True, False],
        'created_at': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'])
    }


def create_general_sample_data() -> dict:
    """Create general sample data"""
    return {
        'item_id': [1.0, 2.0, 3.0, 4.0, 5.0],
        'quantity': [10.0, 20.0, 15.0, 30.0, 25.0],
        'price': [19.99, 29.99, 15.50, 45.00, 35.25],
        'category': ['A', 'B', 'A', 'C', 'B'],
        'in_stock': [True, True, False, True, True]
    }


root_agent = Agent(
    name="post_export_etl_agent",
    model="gemini-2.0-flash",
    description="Post-Export ETL Agent for AppsFlyer Data Locker customers - processes parquet files with custom transformations",
    instruction="""
    You are a specialized ETL (Extract, Transform, Load) agent designed for AppsFlyer Data Locker customers. 
    Your primary function is to process parquet files exported from AppsFlyer's Data Locker and apply 
    custom transformations based on user requirements.

    **Available Tools - ALL RETURN DICTIONARIES:**
    1. `process_parquet_file(file_path, transformations)` - Process parquet files with transformations
    2. `analyze_parquet_structure(file_path)` - Analyze file structure and data types  
    3. `create_sample_parquet(output_path, data_type)` - Create sample data for testing

    **When to Use Each Tool:**

    **Use process_parquet_file when:**
    - User says "I want to get all numbers as type int"
    - User requests data transformations
    - User wants to convert data types
    - User asks to process a specific file

    **Use analyze_parquet_structure when:**
    - User asks "what's in this file?"
    - User wants to understand data structure
    - User asks about data types or columns
    - User needs file analysis before processing

    **Use create_sample_parquet when:**
    - User needs test data
    - User asks to create sample files
    - User wants to test ETL operations
    - No existing file is available

    **Response Pattern:**
    Always call the appropriate tool first, then interpret the results for the user in a friendly way.

    **Examples:**
    - User: "I want to get all numbers as type int from data.parquet"
      → Call: process_parquet_file("data.parquet", "convert_numbers_to_int")
    
    - User: "Analyze my file structure"  
      → Call: analyze_parquet_structure("filename.parquet")
    
    - User: "Create test data"
      → Call: create_sample_parquet("test.parquet", "appsflyer")

    **Important:**
    - All tools return dictionaries with status, results, and detailed information
    - Always check the "status" field in tool responses
    - Provide clear explanations of what the tools found/did
    - Suggest next steps based on tool results
    """,
    tools=[process_parquet_file, analyze_parquet_structure, create_sample_parquet],
)
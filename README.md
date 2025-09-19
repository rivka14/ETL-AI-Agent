# ETL AI Agent

A sophisticated ETL (Extract, Transform, Load) AI Agent system built with Google's Agent Development Kit (ADK) 
for automated BigQuery data operations. 
This project provides multiple specialized AI agents that can handle various aspects of data processing, 
from file uploads to complex data analysis and transformations.

## 🚀 Overview

ETL AI Agent is designed to streamline BigQuery operations through intelligent automation. 
The system consists of three specialized agents, each optimized for specific data tasks:

- **Root Agent**: General-purpose BigQuery analyst for data analysis and transformations
- **Handle File Agent**: Specialized file upload agent for CSV and Parquet files
- **MCP Agent**: Advanced BigQuery analysis agent using Model Context Protocol (MCP)

## 🏗️ Architecture

### Agent Components

#### 1. Root Agent (`root_agent/`)
- **Purpose**: Intelligent BigQuery analyst with dynamic Python script execution
- **Model**: Gemini 2.5 Flash
- **Capabilities**:
  - Schema exploration and data sampling
  - Dynamic Python script execution for complex transformations
  - BigQuery query generation and optimization
  - Data analysis and reporting

#### 2. Handle File Agent (`handle_file_agent/`)
- **Purpose**: Specialized file upload operations to BigQuery
- **Model**: Gemini 2.5 Flash  
- **Capabilities**:
  - CSV and Parquet file upload to BigQuery
  - Automatic schema detection
  - Data validation and integrity checks
  - Error handling and user feedback

#### 3. MCP Agent (`mcp_agent/`)
- **Purpose**: Advanced data analysis using MCP tools
- **Model**: Gemini 2.5 Flash
- **Capabilities**:
  - Structured workflow execution
  - Advanced query optimization
  - Real-time BigQuery operations
  - Comprehensive error handling and retry logic

## 🛠️ Setup and Configuration

### Prerequisites

- Python 3.8+
- Google Cloud Platform account with BigQuery access
- Google ADK (Agent Development Kit)
- Required Python packages (see requirements below)

### Environment Variables

Create a `.env` file in the project root with the following variables:

```env
BIGQUERY_PROJECT=your-gcp-project-id
BIGQUERY_DATASET=your-dataset-name
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account-key.json
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/rivka14/ETL-AI-Agent.git
cd ETL-AI-Agent
```

2. Install dependencies:
```bash
pip install google-cloud-bigquery pandas numpy python-dotenv google-adk
```

3. Set up Google Cloud credentials:
   - Create a service account in your GCP project
   - Download the JSON key file
   - Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable

## 💻 Usage

### Using the Root Agent

The root agent provides comprehensive BigQuery analysis capabilities:

```python
from root_agent.agent import root_agent

# The agent can:
# - Explore table schemas
# - Execute custom Python scripts
# - Perform data transformations
# - Generate insights and reports
```

**Available Tools:**
- `get_schema`: Explore table structure and sample data
- `execute_script`: Run Python code for queries and transformations
- `upload_file_to_bigquery`: Upload files to BigQuery tables

### Using the Handle File Agent

Specialized for file upload operations:

```python
from handle_file_agent.agent import root_agent

# Upload CSV or Parquet files to BigQuery
# Automatic validation and error handling
# Support for various file formats
```

**Key Features:**
- Automatic file format detection (CSV/Parquet)
- Schema auto-detection
- Data validation before upload
- Comprehensive error reporting
- Table naming validation

### Using the MCP Agent

Advanced analysis with structured workflows:

```python
from mcp_agent.agent import root_agent

# Structured 4-step workflow:
# 1. Input Validation
# 2. Query Execution  
# 3. Data Processing & Analysis
# 4. Response Formatting
```

**Workflow Steps:**
1. **Input Validation**: Parameter validation and safety checks
2. **Query Execution**: Optimized BigQuery operations
3. **Data Processing**: Analysis and aggregations
4. **Response Formatting**: Structured output with insights

## 📁 Project Structure

```
ETL-AI-Agent/
├── root_agent/
│   ├── __init__.py
│   ├── agent.py          # Main root agent definition
│   └── tools.py          # BigQuery tools and utilities
├── handle_file_agent/
│   ├── __init__.py
│   ├── agent.py          # File handling agent definition
│   ├── tools.py          # File upload utilities
│   ├── file1.csv         # Sample CSV file
│   └── file2.parquet     # Sample Parquet file
├── mcp_agent/
│   ├── __init__.py
│   └── agent.py          # MCP-based agent definition
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## 🔧 Configuration Details

### Default Table Configuration
- **Project**: `platform-hackaton-2025`
- **Dataset**: `mock_report_Data_Locker`
- **Table**: `mock_report`

### Supported File Formats
- CSV files (`.csv`)
- Parquet files (`.parquet`)

### Security Features
- Environment variable validation
- Service account authentication
- SQL injection prevention through parameterized queries
- File type validation
- Table name sanitization

## 🚨 Error Handling

Each agent includes comprehensive error handling:

- **Connection Errors**: Automatic retry logic (up to 3 attempts)
- **Query Errors**: Detailed error messages with suggestions
- **File Errors**: Validation and format checking
- **Permission Errors**: Clear guidance on access requirements
- **Timeout Errors**: Query optimization suggestions

## 📊 Output Format

All agents provide structured responses:

```json
{
  "status": "success|error",
  "data": "...",          // For successful operations
  "error": "...",         // For error cases
  "insights": "...",      // Analysis results
  "metadata": "..."       // Additional context
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
- Open an issue on GitHub
- Check the documentation for common solutions
- Review error messages for troubleshooting guidance

## 🔮 Future Enhancements

- Support for additional file formats (JSON, XML, Excel)
- Real-time data streaming capabilities
- Advanced data quality validation
- Integration with more cloud platforms
- Enhanced visualization and reporting features

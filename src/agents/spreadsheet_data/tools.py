# src/tools/spreadsheet_tools.py

import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, Any, Optional, List, Union, \
  Tuple  # Keep Union for internal use if needed, but not in tool sigs
from pathlib import Path
import json  # Added import for json dump in get_spreadsheet_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Helper Function ---
def _read_spreadsheet_file(file_path: str, sheet_name_param: Optional[str] = None) -> Tuple[
  Optional[pd.DataFrame], str, Optional[Union[str, int]]]:
  """
    Reads an Excel or CSV file into a pandas DataFrame.
    Handles sheet name as string (attempts conversion to int if numeric).

    Args:
        file_path (str): Absolute path to the spreadsheet file.
        sheet_name_param (Optional[str]): Specific sheet name or index (as a string) to read.
                                          If None or empty, defaults to the first sheet (index 0).
                                          If numeric string (e.g., "0", "1"), tries to use as index.

    Returns:
        Tuple[Optional[pd.DataFrame], str, Optional[Union[str,int]]]: DataFrame, status message, actual sheet name/index used.
    """
  try:
    path = Path(file_path)
    if not path.exists():
      return None, f"File not found: {file_path}", sheet_name_param
    if not path.is_file():
      return None, f"Path is not a file: {file_path}", sheet_name_param

    ext = path.suffix.lower()
    logger.info(f"Reading spreadsheet '{file_path}' with extension '{ext}', requested sheet: {sheet_name_param}")

    sheet_to_read: Optional[Union[str, int]] = None
    actual_sheet_identifier: Optional[Union[str, int]] = None  # Store what was actually used

    if ext in ['.xlsx', '.xls', '.xlsm', '.xlsb', '.odf', '.ods', '.odt']:
      excel_file = pd.ExcelFile(file_path)
      available_sheets = excel_file.sheet_names

      if sheet_name_param:
        # Try converting to int if it looks like a number
        try:
          sheet_index = int(sheet_name_param)
          if 0 <= sheet_index < len(available_sheets):
            sheet_to_read = sheet_index
            actual_sheet_identifier = sheet_index
            logger.info(f"Using sheet index: {sheet_index}")
          else:
            logger.warning(
              f"Sheet index '{sheet_name_param}' out of range. Using first sheet (0). Available: {available_sheets}")
            sheet_to_read = 0
            actual_sheet_identifier = 0
        except ValueError:
          # Not an integer, treat as sheet name
          if sheet_name_param in available_sheets:
            sheet_to_read = sheet_name_param
            actual_sheet_identifier = sheet_name_param
            logger.info(f"Using sheet name: {sheet_name_param}")
          else:
            logger.warning(
              f"Sheet name '{sheet_name_param}' not found. Using first sheet '{available_sheets[0]}'. Available: {available_sheets}")
            sheet_to_read = available_sheets[0]  # Use first sheet name
            actual_sheet_identifier = available_sheets[0]
      else:
        # Default to first sheet (index 0)
        sheet_to_read = 0
        actual_sheet_identifier = 0
        logger.info("No sheet specified, using first sheet (index 0).")

      df = pd.read_excel(file_path, sheet_name=sheet_to_read)
      logger.info(f"Successfully read Excel file '{file_path}', sheet '{actual_sheet_identifier}'")
      return df, "success", actual_sheet_identifier

    elif ext == '.csv':
      df = pd.read_csv(file_path)
      logger.info(f"Successfully read CSV file '{file_path}'")
      return df, "success", None  # CSVs don't have sheets
    elif ext == '.tsv':
      df = pd.read_csv(file_path, sep='\t')
      logger.info(f"Successfully read TSV file '{file_path}'")
      return df, "success", None
    else:
      logger.warning(f"Unsupported spreadsheet extension: {ext} for file {file_path}")
      return None, f"Unsupported file extension: {ext}. Supported: .xlsx, .xls, .csv, .tsv, etc.", sheet_name_param

  except FileNotFoundError:
    logger.error(f"File not found: {file_path}")
    return None, f"File not found: {file_path}", sheet_name_param
  except Exception as e:
    logger.error(f"Error processing spreadsheet {file_path}: {e}", exc_info=True)
    return None, f"Error processing spreadsheet: {str(e)}", sheet_name_param


# --- Tool Implementations ---

def get_spreadsheet_info(file_path: str, sheet_name: Optional[str] = None) -> Dict[str, Any]:  # Changed sheet_name type
  """
    Provides detailed metadata about a spreadsheet file.
    Includes shape, columns, data types, missing values, basic numeric stats, and sample rows.

    Args:
        file_path (str): Absolute path to the spreadsheet file (.xlsx, .xls, .csv, etc.).
        sheet_name (Optional[str]): Specific sheet name or index (as string, e.g., "Sheet1", "0"). Defaults to first sheet.

    Returns:
        dict: Status and a dictionary containing the spreadsheet information or an error message.
    """
  logger.info(f"Getting info for spreadsheet: {file_path}, requested sheet: {sheet_name}")
  df, status_msg, actual_sheet = _read_spreadsheet_file(file_path, sheet_name)  # Pass sheet_name as str
  if df is None:
    return {"status": "error", "message": status_msg}

  try:
    info = {
      "file_path": file_path,
      "sheet_analyzed": actual_sheet,  # Report the sheet actually used
      "shape": {"rows": df.shape[0], "columns": df.shape[1]},
      "columns": list(df.columns),
      "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
      "missing_values_count": {col: int(count) for col, count in df.isnull().sum().items() if count > 0},
      "total_memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
      "sample_head": df.head(3).to_dict(orient='records'),
      "sample_tail": df.tail(3).to_dict(orient='records'),
    }

    numeric_cols = df.select_dtypes(include=np.number).columns
    if not numeric_cols.empty:
      stats = df[numeric_cols].describe().round(2).to_dict()
      info["numeric_column_stats"] = stats

    logger.info(f"Successfully retrieved info for {file_path}")
    return {"status": "success", "info": json.dumps(info, indent=2, default=str)}
  except Exception as e:
    logger.error(f"Error generating info for {file_path}: {e}", exc_info=True)
    return {"status": "error", "message": f"Error generating spreadsheet info: {str(e)}"}


def get_sheet_names(file_path: str) -> Dict[str, Any]:
  """
    Lists the names of all sheets in an Excel workbook.

    Args:
        file_path (str): Absolute path to the Excel file (.xlsx, .xls, etc.).

    Returns:
        dict: Status and a list of sheet names or an error message.
    """
  logger.info(f"Getting sheet names for: {file_path}")
  path = Path(file_path)
  ext = path.suffix.lower()

  if ext not in ['.xlsx', '.xls', '.xlsm', '.xlsb', '.odf', '.ods', '.odt']:
    return {"status": "error", "message": f"Not an Excel file: {file_path}. Cannot get sheet names."}

  try:
    excel_file = pd.ExcelFile(file_path)
    sheet_names_list = excel_file.sheet_names
    logger.info(f"Successfully retrieved sheet names for {file_path}: {sheet_names_list}")
    return {"status": "success", "sheet_names": sheet_names_list}
  except FileNotFoundError:
    logger.error(f"File not found: {file_path}")
    return {"status": "error", "message": f"File not found: {file_path}"}
  except Exception as e:
    logger.error(f"Error getting sheet names for {file_path}: {e}", exc_info=True)
    return {"status": "error", "message": f"Error reading sheet names: {str(e)}"}


def get_cell_value(file_path: str, cell_coordinate: str, sheet_name: Optional[str] = None) -> Dict[
  str, Any]:  # Changed sheet_name type
  """
    Retrieves the value of a specific cell in a spreadsheet.

    Args:
        file_path (str): Absolute path to the spreadsheet file.
        cell_coordinate (str): The cell address (e.g., "A1", "C5").
        sheet_name (Optional[str]): Specific sheet name or index (as string). Defaults to the first sheet.

    Returns:
        dict: Status and the cell value or an error message.
    """
  logger.info(f"Getting cell value for: {file_path}, requested sheet: {sheet_name}, cell: {cell_coordinate}")
  df, status_msg, _ = _read_spreadsheet_file(file_path, sheet_name)  # Pass sheet_name as str
  if df is None:
    return {"status": "error", "message": status_msg}

  try:
    col_str = ''.join(filter(str.isalpha, cell_coordinate))
    row_str = ''.join(filter(str.isdigit, cell_coordinate))
    if not col_str or not row_str: raise ValueError("Invalid format")
    row_index = int(row_str) - 1
    col_index = 0
    for char in col_str.upper(): col_index = col_index * 26 + (ord(char) - ord('A') + 1)
    col_index -= 1

    if row_index < 0 or row_index >= df.shape[0] or col_index < 0 or col_index >= df.shape[1]:
      logger.error(f"Cell coordinate {cell_coordinate} ({row_index}, {col_index}) out of bounds for shape {df.shape}")
      return {"status": "error", "message": f"Cell coordinate {cell_coordinate} is out of bounds."}

    value = df.iloc[row_index, col_index]
    if pd.isna(value):
      value = None
    elif isinstance(value, (np.int64, np.int32)):
      value = int(value)
    elif isinstance(value, (np.float64, np.float32)):
      value = float(value)
    elif isinstance(value, pd.Timestamp):
      value = value.isoformat()

    logger.info(f"Successfully retrieved value from {cell_coordinate}: {value}")
    return {"status": "success", "value": value}
  except ValueError:
    logger.error(f"Invalid cell coordinate format: {cell_coordinate}")
    return {"status": "error", "message": f"Invalid cell coordinate format: {cell_coordinate}"}
  except Exception as e:
    logger.error(f"Error getting cell value for {cell_coordinate}: {e}", exc_info=True)
    return {"status": "error", "message": f"Error getting cell value: {str(e)}"}


def query_spreadsheet(file_path: str, query_string: str, sheet_name: Optional[str] = None) -> Dict[
  str, Any]:  # Changed sheet_name type
  """
    Filters spreadsheet data based on a pandas query string.

    Args:
        file_path (str): Absolute path to the spreadsheet file.
        query_string (str): The pandas query string (e.g., "`Column Name` > 100").
        sheet_name (Optional[str]): Specific sheet name or index (as string). Defaults to first sheet.

    Returns:
        dict: Status and the filtered data as a markdown string or an error message.
    """
  logger.info(f"Querying spreadsheet: {file_path}, requested sheet: {sheet_name}, query: '{query_string}'")
  df, status_msg, _ = _read_spreadsheet_file(file_path, sheet_name)  # Pass sheet_name as str
  if df is None:
    return {"status": "error", "message": status_msg}

  try:
    logger.info(f"Executing pandas query: {query_string}")
    filtered_df = df.query(query_string, engine='python')

    if filtered_df.empty:
      logger.info("Query returned no results.")
      return {"status": "success", "filtered_data": "Query returned no matching rows."}

    logger.info(f"Query successful, found {len(filtered_df)} matching rows.")
    max_rows, max_cols, max_len = 20, 10, 15000
    display_df = filtered_df.head(max_rows).iloc[:, :max_cols]
    content = display_df.to_markdown(index=False)
    trunc_msgs = []
    if len(filtered_df) > max_rows: trunc_msgs.append(f"showing first {max_rows} of {len(filtered_df)} rows")
    if len(filtered_df.columns) > max_cols: trunc_msgs.append(
      f"showing first {max_cols} of {len(filtered_df.columns)} columns")
    if trunc_msgs: content += f"\n... ({', '.join(trunc_msgs)} truncated)"
    if len(content) > max_len:
      logger.warning("Query result markdown output truncated.")
      content = content[:max_len] + "\n... (output truncated)"

    return {"status": "success", "filtered_data": content}
  except Exception as e:
    logger.error(f"Error executing query '{query_string}': {e}", exc_info=True)
    return {"status": "error", "message": f"Error executing query: {str(e)}"}  # Simplified error message


def calculate_column_stat(file_path: str, column_name: str, stat_type: str, sheet_name: Optional[str] = None) -> Dict[
  str, Any]:  # Changed sheet_name type
  """
    Calculates a specific statistic for a given column in a spreadsheet.

    Args:
        file_path (str): Absolute path to the spreadsheet file.
        column_name (str): The name of the column to analyze.
        stat_type (str): Statistic type (e.g., "sum", "mean", "median", "std").
        sheet_name (Optional[str]): Specific sheet name or index (as string). Defaults to first sheet.

    Returns:
        dict: Status and the calculated result or an error message.
    """
  logger.info(
    f"Calculating stat '{stat_type}' for column '{column_name}' in: {file_path}, requested sheet: {sheet_name}")
  df, status_msg, _ = _read_spreadsheet_file(file_path, sheet_name)  # Pass sheet_name as str
  if df is None:
    return {"status": "error", "message": status_msg}

  if column_name not in df.columns:
    logger.error(f"Column '{column_name}' not found. Available: {list(df.columns)}")
    return {"status": "error", "message": f"Column '{column_name}' not found."}

  column_data = df[column_name]
  stat_type_lower = stat_type.lower().strip()

  stat_functions = {  # Using lambdas for consistency
    "sum": lambda col: col.sum(), "mean": lambda col: col.mean(), "average": lambda col: col.mean(),
    "median": lambda col: col.median(), "min": lambda col: col.min(), "max": lambda col: col.max(),
    "count": lambda col: col.count(), "nunique": lambda col: col.nunique(),
    "std": lambda col: col.std(ddof=1), "var": lambda col: col.var(ddof=1),
    "pstdev": lambda col: col.std(ddof=0), "pvariance": lambda col: col.var(ddof=0),
  }

  if stat_type_lower not in stat_functions:
    logger.error(f"Unsupported statistic type: '{stat_type}'. Supported: {list(stat_functions.keys())}")
    return {"status": "error", "message": f"Unsupported statistic type: '{stat_type}'"}

  try:
    numeric_stats = ["sum", "mean", "average", "median", "min", "max", "std", "var", "pstdev", "pvariance"]
    target_column = column_data

    # Attempt conversion to numeric only if a numeric stat is requested
    if stat_type_lower in numeric_stats and not pd.api.types.is_numeric_dtype(target_column):
      numeric_column_data = pd.to_numeric(target_column, errors='coerce')
      if numeric_column_data.isnull().all():
        logger.warning(f"Column '{column_name}' could not be treated as numeric for statistic '{stat_type}'")
        return {"status": "error", "message": f"Column '{column_name}' is not numeric, cannot calculate '{stat_type}'."}
      else:
        logger.warning(
          f"Column '{column_name}' contained non-numeric values; they were ignored for statistic '{stat_type}'.")
        target_column = numeric_column_data  # Use the converted column for calculation

    # Check for insufficient data for sample std/var
    if stat_type_lower in ["std", "var"] and target_column.count() < 2:
      logger.warning(
        f"Cannot calculate sample {stat_type} for column '{column_name}' with less than 2 valid data points.")
      return {"status": "error", "message": f"Sample {stat_type} requires at least 2 data points."}

    result = stat_functions[stat_type_lower](target_column)

    if pd.isna(result):  # Handle NaN results more broadly
      result = None
      logger.warning(f"Statistic '{stat_type}' resulted in NaN/NaT (possibly due to data type or insufficient data).")
    elif isinstance(result, (np.int64, np.int32)):
      result = int(result)
    elif isinstance(result, (np.float64, np.float32)):
      result = float(result)
    elif isinstance(result, pd.Timestamp):
      result = result.isoformat()

    logger.info(f"Successfully calculated '{stat_type}' for column '{column_name}': {result}")
    return {"status": "success", "result": result}

  except Exception as e:
    logger.error(f"Error calculating statistic '{stat_type}' for column '{column_name}': {e}", exc_info=True)
    return {"status": "error", "message": f"Error calculating statistic '{stat_type}': {str(e)}"}
"""
Data Loader Module for File Data Analyzer System
Handles reading CSV, XLSX, and XLS datasets, extracting metadata,
and validating input formats with robust fallback encodings.
"""

import os
import io
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional, List


class DataLoader:
    """Class responsible for loading and validating dataset files."""

    @staticmethod
    def load_file(file_obj, filename: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Loads a dataframe from an uploaded file object or file path.
        
        Args:
            file_obj: Streamlit UploadedFile, file-like object, or filepath string.
            filename: The name of the file including its extension.
            
        Returns:
            Tuple of (pd.DataFrame or None, error_message or None)
        """
        try:
            ext = os.path.splitext(filename)[1].lower()
            if ext == ".csv":
                # Try UTF-8 first, fallback to latin1/cp1252 if decoding fails
                if hasattr(file_obj, "read"):
                    bytes_data = file_obj.read()
                    file_obj.seek(0)
                    for encoding in ["utf-8", "latin1", "cp1252", "iso-8859-1"]:
                        try:
                            df = pd.read_csv(io.BytesIO(bytes_data), encoding=encoding)
                            return df, None
                        except (UnicodeDecodeError, pd.errors.ParserError):
                            continue
                    return None, "Failed to decode CSV file with standard encodings (UTF-8, Latin-1, CP1252)."
                else:
                    df = pd.read_csv(file_obj)
                    return df, None

            elif ext in [".xlsx", ".xls"]:
                df = pd.read_excel(file_obj)
                return df, None
            else:
                return None, f"Unsupported file extension '{ext}'. Please upload a .csv, .xlsx, or .xls file."

        except pd.errors.EmptyDataError:
            return None, "The uploaded file is empty. Please provide a valid dataset with data."
        except Exception as e:
            return None, f"An unexpected error occurred while loading the file: {str(e)}"

    @staticmethod
    def load_sample_dataset() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Loads the default sample sales dataset included with the project."""
        sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_sales_data.csv")
        if os.path.exists(sample_path):
            try:
                df = pd.read_csv(sample_path)
                return df, None
            except Exception as e:
                return None, f"Failed to load sample dataset: {str(e)}"
        return None, "Sample dataset file 'data/sample_sales_data.csv' not found."

    @staticmethod
    def get_dataset_metadata(df: pd.DataFrame, filename: str, file_size_bytes: int = 0) -> Dict[str, Any]:
        """
        Extracts high-level summary metadata from a DataFrame.
        
        Args:
            df: The loaded pandas DataFrame.
            filename: Name of the dataset file.
            file_size_bytes: Optional size in bytes.
            
        Returns:
            Dictionary containing metadata and column taxonomy.
        """
        if df is None or df.empty:
            return {
                "filename": filename,
                "file_size": "0 KB",
                "rows": 0,
                "columns": 0,
                "memory_usage": "0 KB",
                "numerical_cols": [],
                "categorical_cols": [],
                "datetime_cols": [],
                "missing_values": 0,
                "duplicate_rows": 0,
            }

        # Calculate memory usage
        mem_bytes = df.memory_usage(deep=True).sum()
        if mem_bytes < 1024 * 1024:
            mem_str = f"{mem_bytes / 1024:.2f} KB"
        else:
            mem_str = f"{mem_bytes / (1024 * 1024):.2f} MB"

        # Calculate file size string
        if file_size_bytes <= 0:
            file_size_bytes = mem_bytes
        if file_size_bytes < 1024 * 1024:
            file_size_str = f"{file_size_bytes / 1024:.2f} KB"
        else:
            file_size_str = f"{file_size_bytes / (1024 * 1024):.2f} MB"

        # Classify columns
        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        cat_cols = list(df.select_dtypes(include=["object", "category", "bool"]).columns)
        date_cols = list(df.select_dtypes(include=["datetime", "datetime64"]).columns)

        # Detect prospective date columns among object columns
        prospective_date_cols = []
        for col in cat_cols:
            col_lower = col.lower()
            if any(k in col_lower for k in ["date", "time", "year", "timestamp"]):
                prospective_date_cols.append(col)

        missing_count = int(df.isna().sum().sum())
        duplicate_count = int(df.duplicated().sum())

        return {
            "filename": filename,
            "file_size": file_size_str,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "memory_usage": mem_str,
            "numerical_cols": num_cols,
            "categorical_cols": cat_cols,
            "datetime_cols": date_cols,
            "prospective_date_cols": prospective_date_cols,
            "missing_values": missing_count,
            "duplicate_rows": duplicate_count,
        }

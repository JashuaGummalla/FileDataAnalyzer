"""
Data Cleaner Module for File Data Analyzer System
Provides data preprocessing tools: duplicate removal, missing value imputation,
column renaming, type conversions, and IQR filtering.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional


class DataCleaner:
    """Provides methods for cleaning and transforming pandas DataFrames."""

    @staticmethod
    def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> Tuple[pd.DataFrame, int]:
        """
        Removes duplicate rows from the dataset.
        
        Args:
            df: The pandas DataFrame.
            subset: Optional list of columns to consider for identifying duplicates.
            
        Returns:
            Tuple of (cleaned_df, count_of_removed_rows)
        """
        initial_count = len(df)
        cleaned_df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
        removed = initial_count - len(cleaned_df)
        return cleaned_df, removed

    @staticmethod
    def handle_missing_values(
        df: pd.DataFrame,
        strategy: str,
        columns: Optional[List[str]] = None,
        custom_fill: Optional[Any] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Handles missing values in the dataset using various statistical imputation or drop techniques.
        
        Args:
            df: The pandas DataFrame.
            strategy: 'drop_rows', 'fill_mean', 'fill_median', 'fill_mode', 'fill_zero', 'fill_custom'
            columns: Specific columns to apply the strategy to (if None, applies to all compatible columns).
            custom_fill: Custom value when strategy is 'fill_custom'.
            
        Returns:
            Tuple of (cleaned_df, summary_report)
        """
        cleaned_df = df.copy()
        target_cols = columns if columns is not None and len(columns) > 0 else list(cleaned_df.columns)
        initial_missing = int(cleaned_df[target_cols].isna().sum().sum())
        initial_rows = len(cleaned_df)
        log = []

        if strategy == "drop_rows":
            cleaned_df = cleaned_df.dropna(subset=target_cols).reset_index(drop=True)
            rows_dropped = initial_rows - len(cleaned_df)
            log.append(f"Dropped {rows_dropped} rows containing missing values in {target_cols}.")

        elif strategy == "fill_mean":
            num_cols = cleaned_df[target_cols].select_dtypes(include=[np.number]).columns
            for col in num_cols:
                if cleaned_df[col].isna().sum() > 0:
                    col_vals = cleaned_df[col].dropna().values
                    if len(col_vals) > 0:
                        # Use NumPy to calculate mean explicitly
                        mean_val = float(np.mean(col_vals))
                        cleaned_df[col] = cleaned_df[col].fillna(mean_val)
                        log.append(f"Filled missing values in '{col}' with NumPy mean: {mean_val:.2f}")

        elif strategy == "fill_median":
            num_cols = cleaned_df[target_cols].select_dtypes(include=[np.number]).columns
            for col in num_cols:
                if cleaned_df[col].isna().sum() > 0:
                    col_vals = cleaned_df[col].dropna().values
                    if len(col_vals) > 0:
                        # Use NumPy to calculate median explicitly
                        median_val = float(np.median(col_vals))
                        cleaned_df[col] = cleaned_df[col].fillna(median_val)
                        log.append(f"Filled missing values in '{col}' with NumPy median: {median_val:.2f}")

        elif strategy == "fill_mode":
            for col in target_cols:
                if cleaned_df[col].isna().sum() > 0:
                    mode_series = cleaned_df[col].mode()
                    if not mode_series.empty:
                        mode_val = mode_series.iloc[0]
                        cleaned_df[col] = cleaned_df[col].fillna(mode_val)
                        log.append(f"Filled missing values in '{col}' with mode: '{mode_val}'")

        elif strategy == "fill_zero":
            num_cols = cleaned_df[target_cols].select_dtypes(include=[np.number]).columns
            for col in num_cols:
                if cleaned_df[col].isna().sum() > 0:
                    cleaned_df[col] = cleaned_df[col].fillna(0)
                    log.append(f"Filled missing values in '{col}' with 0")

        elif strategy == "fill_custom":
            fill_val = custom_fill if custom_fill is not None else "Unknown"
            for col in target_cols:
                if cleaned_df[col].isna().sum() > 0:
                    cleaned_df[col] = cleaned_df[col].fillna(fill_val)
                    log.append(f"Filled missing values in '{col}' with custom value: '{fill_val}'")

        remaining_missing = int(cleaned_df[target_cols].isna().sum().sum())
        imputed_count = initial_missing - remaining_missing

        summary = {
            "strategy": strategy,
            "columns_affected": target_cols,
            "initial_missing": initial_missing,
            "remaining_missing": remaining_missing,
            "imputed_or_dropped_count": imputed_count,
            "logs": log,
        }
        return cleaned_df, summary

    @staticmethod
    def standardize_column_names(df: pd.DataFrame, style: str = "snake_case") -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Cleans and standardizes column headers.
        
        Args:
            df: The pandas DataFrame.
            style: 'snake_case', 'lowercase', 'uppercase', 'strip_spaces'
            
        Returns:
            Tuple of (renamed_df, rename_mapping)
        """
        cleaned_df = df.copy()
        mapping = {}

        for col in cleaned_df.columns:
            new_col = str(col)
            if style == "snake_case":
                new_col = new_col.strip().replace(" ", "_").replace("-", "_").replace(".", "_").lower()
                while "__" in new_col:
                    new_col = new_col.replace("__", "_")
            elif style == "lowercase":
                new_col = new_col.strip().lower()
            elif style == "uppercase":
                new_col = new_col.strip().upper()
            elif style == "strip_spaces":
                new_col = new_col.strip()
            
            mapping[col] = new_col

        cleaned_df = cleaned_df.rename(columns=mapping)
        return cleaned_df, mapping

    @staticmethod
    def rename_single_column(df: pd.DataFrame, old_name: str, new_name: str) -> Tuple[pd.DataFrame, Optional[str]]:
        """Renames a single column with validation."""
        if old_name not in df.columns:
            return df, f"Column '{old_name}' not found."
        new_name_clean = new_name.strip()
        if not new_name_clean:
            return df, "New column name cannot be empty."
        if new_name_clean in df.columns and new_name_clean != old_name:
            return df, f"A column named '{new_name_clean}' already exists."

        cleaned_df = df.rename(columns={old_name: new_name_clean})
        return cleaned_df, None

    @staticmethod
    def convert_column_type(df: pd.DataFrame, column: str, target_type: str) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        Safely converts a column's data type.
        
        Args:
            df: The pandas DataFrame.
            column: Name of column to cast.
            target_type: 'integer', 'float', 'string', 'datetime', 'category', 'boolean'
            
        Returns:
            Tuple of (converted_df, error_or_success_message)
        """
        if column not in df.columns:
            return df, f"Column '{column}' does not exist."

        cleaned_df = df.copy()
        try:
            if target_type == "integer":
                cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors="coerce").fillna(0).astype(np.int64)
            elif target_type == "float":
                cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors="coerce").astype(np.float64)
            elif target_type == "string":
                cleaned_df[column] = cleaned_df[column].astype(str)
            elif target_type == "datetime":
                cleaned_df[column] = pd.to_datetime(cleaned_df[column], errors="coerce")
            elif target_type == "category":
                cleaned_df[column] = cleaned_df[column].astype("category")
            elif target_type == "boolean":
                cleaned_df[column] = cleaned_df[column].astype(bool)
            else:
                return df, f"Unsupported target type: '{target_type}'"

            return cleaned_df, None
        except Exception as e:
            return df, f"Failed to convert column '{column}' to {target_type}: {str(e)}"

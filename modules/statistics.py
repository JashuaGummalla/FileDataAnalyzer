"""
Statistical Analysis Module for File Data Analyzer System
Demonstrates practical numerical computing with NumPy and Pandas.
Implements descriptive statistics, IQR outlier detection, correlation analysis,
and data quality scoring.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


class StatisticalAnalyzer:
    """Computes comprehensive statistical metrics using NumPy and Pandas."""

    @staticmethod
    def compute_numerical_summary(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates detailed descriptive statistics for all numerical columns
        explicitly utilizing NumPy vectorized operations.
        
        Args:
            df: The pandas DataFrame.
            
        Returns:
            DataFrame with statistical metrics as rows and column names as columns.
        """
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            return pd.DataFrame()

        metrics = {}
        for col in num_cols:
            clean_series = df[col].dropna()
            arr = clean_series.to_numpy(dtype=np.float64)

            if len(arr) == 0:
                continue

            # Explicit NumPy computations
            count_val = len(arr)
            mean_val = float(np.mean(arr))
            median_val = float(np.median(arr))
            std_val = float(np.std(arr, ddof=1)) if count_val > 1 else 0.0
            var_val = float(np.var(arr, ddof=1)) if count_val > 1 else 0.0
            min_val = float(np.min(arr))
            max_val = float(np.max(arr))
            range_val = float(np.ptp(arr))  # Peak-to-peak range using NumPy
            q25 = float(np.percentile(arr, 25))
            q50 = float(np.percentile(arr, 50))
            q75 = float(np.percentile(arr, 75))
            iqr_val = q75 - q25

            # Mode from pandas
            mode_s = clean_series.mode()
            mode_val = float(mode_s.iloc[0]) if not mode_s.empty else np.nan

            # Skewness and Kurtosis
            skew_val = float(clean_series.skew()) if count_val > 2 else 0.0
            kurt_val = float(clean_series.kurtosis()) if count_val > 3 else 0.0

            metrics[col] = {
                "Count": count_val,
                "Mean": round(mean_val, 4),
                "Median (50th %)": round(median_val, 4),
                "Mode": round(mode_val, 4) if not np.isnan(mode_val) else "N/A",
                "Standard Deviation": round(std_val, 4),
                "Variance": round(var_val, 4),
                "Minimum": round(min_val, 4),
                "Maximum": round(max_val, 4),
                "Range (Max - Min)": round(range_val, 4),
                "25th Percentile (Q1)": round(q25, 4),
                "50th Percentile (Q2)": round(q50, 4),
                "75th Percentile (Q3)": round(q75, 4),
                "IQR (Q3 - Q1)": round(iqr_val, 4),
                "Skewness": round(skew_val, 4),
                "Kurtosis": round(kurt_val, 4),
            }

        return pd.DataFrame(metrics)

    @staticmethod
    def get_column_numpy_inspection(df: pd.DataFrame, col: str) -> Dict[str, Any]:
        """
        Detailed single-column NumPy breakdown for lab viva demonstration.
        Shows exact NumPy functions called and their raw scalar outputs.
        """
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            return {}

        clean_series = df[col].dropna()
        arr = clean_series.to_numpy(dtype=np.float64)

        if len(arr) == 0:
            return {}

        return {
            "Array Shape": arr.shape,
            "Array Data Type": str(arr.dtype),
            "np.mean()": float(np.mean(arr)),
            "np.median()": float(np.median(arr)),
            "np.std()": float(np.std(arr)),
            "np.var()": float(np.var(arr)),
            "np.min()": float(np.min(arr)),
            "np.max()": float(np.max(arr)),
            "np.ptp() (Range)": float(np.ptp(arr)),
            "np.percentile(25)": float(np.percentile(arr, 25)),
            "np.percentile(75)": float(np.percentile(arr, 75)),
            "np.sum()": float(np.sum(arr)),
            "Zero Count": int(np.count_nonzero(arr == 0)),
            "Positive Count": int(np.count_nonzero(arr > 0)),
            "Negative Count": int(np.count_nonzero(arr < 0)),
        }

    @staticmethod
    def compute_categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes summary statistics for categorical and text columns.
        """
        cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
        if len(cat_cols) == 0:
            return pd.DataFrame()

        rows = []
        for col in cat_cols:
            s = df[col]
            unique_cnt = s.nunique(dropna=True)
            mode_s = s.mode()
            top_val = mode_s.iloc[0] if not mode_s.empty else "N/A"
            top_freq = int(s.value_counts().iloc[0]) if unique_cnt > 0 else 0
            top_pct = round((top_freq / len(s)) * 100, 2) if len(s) > 0 else 0.0
            missing_cnt = int(s.isna().sum())

            rows.append({
                "Column": col,
                "Data Type": str(s.dtype),
                "Unique Count": unique_cnt,
                "Most Frequent (Mode)": str(top_val),
                "Mode Frequency": top_freq,
                "Mode % of Total": f"{top_pct}%",
                "Missing Values": missing_cnt,
            })

        return pd.DataFrame(rows)

    @staticmethod
    def detect_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs robust Outlier Detection using the Interquartile Range (IQR) method.
        
        Q1 = 25th percentile
        Q3 = 75th percentile
        IQR = Q3 - Q1
        Lower Bound = Q1 - 1.5 * IQR
        Upper Bound = Q3 + 1.5 * IQR
        """
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            return pd.DataFrame()

        outlier_data = []
        for col in num_cols:
            series = df[col].dropna()
            arr = series.to_numpy(dtype=np.float64)

            if len(arr) < 4:
                continue

            q1 = float(np.percentile(arr, 25))
            q3 = float(np.percentile(arr, 75))
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = arr[(arr < lower_bound) | (arr > upper_bound)]
            outlier_cnt = len(outliers)
            outlier_pct = round((outlier_cnt / len(arr)) * 100, 2) if len(arr) > 0 else 0.0

            outlier_data.append({
                "Column": col,
                "Total Count": len(arr),
                "Q1 (25%)": round(q1, 4),
                "Q3 (75%)": round(q3, 4),
                "IQR": round(iqr, 4),
                "Lower Bound": round(lower_bound, 4),
                "Upper Bound": round(upper_bound, 4),
                "Outlier Count": outlier_cnt,
                "Outlier %": f"{outlier_pct}%",
            })

        return pd.DataFrame(outlier_data)

    @staticmethod
    def calculate_data_quality_score(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates an application-generated Data Quality Score (0 - 100%)
        based on completeness, uniqueness, and consistency.
        """
        if df is None or df.empty:
            return {
                "score": 0.0,
                "grade": "F",
                "missing_penalty": 0.0,
                "duplicate_penalty": 0.0,
                "explanation": "Dataset is empty.",
            }

        total_cells = df.size
        missing_cells = int(df.isna().sum().sum())
        missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0

        total_rows = len(df)
        duplicate_rows = int(df.duplicated().sum())
        duplicate_pct = (duplicate_rows / total_rows) * 100 if total_rows > 0 else 0

        # Penalties (60% weight on missing values, 40% weight on duplicates)
        missing_penalty = missing_pct * 0.6
        duplicate_penalty = duplicate_pct * 0.4
        total_penalty = missing_penalty + duplicate_penalty

        score = max(0.0, min(100.0, 100.0 - total_penalty))

        if score >= 90:
            grade = "Excellent (A)"
        elif score >= 80:
            grade = "Good (B)"
        elif score >= 70:
            grade = "Fair (C)"
        elif score >= 50:
            grade = "Poor (D)"
        else:
            grade = "Critical Issues (F)"

        explanation = (
            f"Quality Score is computed based on data completeness and record uniqueness. "
            f"Formula: 100% - ({missing_pct:.1f}% Missing × 0.6 + {duplicate_pct:.1f}% Duplicates × 0.4)."
        )

        return {
            "score": round(score, 1),
            "grade": grade,
            "missing_pct": round(missing_pct, 2),
            "duplicate_pct": round(duplicate_pct, 2),
            "missing_penalty": round(missing_penalty, 2),
            "duplicate_penalty": round(duplicate_penalty, 2),
            "explanation": explanation,
        }

    @staticmethod
    def compute_correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Computes the correlation matrix for numerical columns and extracts key relationships.
        
        Args:
            df: The pandas DataFrame.
            method: 'pearson' or 'spearman'
            
        Returns:
            Tuple of (correlation_matrix_df, strong_correlations_list)
        """
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) < 2:
            return pd.DataFrame(), []

        corr_matrix = df[num_cols].corr(method=method)

        # Extract top pairs
        pairs = []
        seen = set()
        for col1 in num_cols:
            for col2 in num_cols:
                if col1 != col2 and (col2, col1) not in seen:
                    val = corr_matrix.loc[col1, col2]
                    if not np.isnan(val):
                        seen.add((col1, col2))
                        pairs.append({
                            "Feature 1": col1,
                            "Feature 2": col2,
                            "Correlation": round(float(val), 4),
                            "Strength": (
                                "Very Strong Positive" if val >= 0.8 else
                                "Strong Positive" if val >= 0.5 else
                                "Moderate Positive" if val >= 0.3 else
                                "Very Strong Negative" if val <= -0.8 else
                                "Strong Negative" if val <= -0.5 else
                                "Moderate Negative" if val <= -0.3 else
                                "Weak / Neutral"
                            )
                        })

        # Sort by absolute correlation descending
        pairs.sort(key=lambda x: abs(x["Correlation"]), reverse=True)
        return corr_matrix, pairs

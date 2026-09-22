"""
Automatic Insights Engine for File Data Analyzer System
Generates rule-based, deterministic analytical findings, data hygiene audits,
and dynamic statistical summaries without external AI dependencies.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime


class InsightsEngine:
    """Intelligent rule-based data analysis insights generator."""

    @staticmethod
    def generate_all_insights(df: pd.DataFrame) -> Dict[str, List[Dict[str, str]]]:
        """
        Generates categorized insights across dataset structure, numerical metrics,
        correlations, categories, outliers, and time trends.
        """
        if df is None or df.empty:
            return {
                "health": [{"type": "warning", "text": "Dataset is empty. No insights can be generated."}],
                "numerical": [],
                "correlation": [],
                "categorical": [],
                "outliers": [],
                "time_series": []
            }

        insights = {
            "health": InsightsEngine._dataset_health_insights(df),
            "numerical": InsightsEngine._numerical_insights(df),
            "correlation": InsightsEngine._correlation_insights(df),
            "categorical": InsightsEngine._categorical_insights(df),
            "outliers": InsightsEngine._outlier_insights(df),
            "time_series": InsightsEngine._time_series_insights(df),
        }
        return insights

    @staticmethod
    def _dataset_health_insights(df: pd.DataFrame) -> List[Dict[str, str]]:
        """Analyzes overall dataset dimensions, missingness, and duplicates."""
        items = []
        n_rows, n_cols = len(df), len(df.columns)
        total_cells = df.size
        missing_count = int(df.isna().sum().sum())
        missing_pct = (missing_count / total_cells) * 100 if total_cells > 0 else 0
        dup_count = int(df.duplicated().sum())
        dup_pct = (dup_count / n_rows) * 100 if n_rows > 0 else 0

        # Dimension summary
        items.append({
            "type": "info",
            "title": "Dataset Scale",
            "text": f"The dataset consists of **{n_rows:,} records** across **{n_cols} attributes**, representing a total volume of **{total_cells:,} data cells**."
        })

        # Missing values audit
        cols_with_na = df.columns[df.isna().any()].tolist()
        if missing_count == 0:
            items.append({
                "type": "success",
                "title": "Perfect Completeness",
                "text": "There are zero missing values across all columns. The dataset exhibits 100% data completeness."
            })
        else:
            cols_str = ", ".join([f"**{c}** ({df[c].isna().sum()} nulls)" for c in cols_with_na[:4]])
            items.append({
                "type": "warning",
                "title": "Missing Values Identified",
                "text": f"Found **{missing_count:,} missing values** ({missing_pct:.2f}% of all cells) across **{len(cols_with_na)} column(s)**: {cols_str}."
            })

        # Duplicates audit
        if dup_count == 0:
            items.append({
                "type": "success",
                "title": "Unique Records",
                "text": "No duplicate rows were detected. All records are distinct."
            })
        else:
            items.append({
                "type": "warning",
                "title": "Duplicate Entries Detected",
                "text": f"Detected **{dup_count} duplicate rows** ({dup_pct:.2f}% of dataset). Removing duplicates in the Data Cleaning module is recommended."
            })

        return items

    @staticmethod
    def _numerical_insights(df: pd.DataFrame) -> List[Dict[str, str]]:
        """Calculates distribution, mean, variance, and skewness observations."""
        items = []
        num_cols = df.select_dtypes(include=[np.number]).columns

        if len(num_cols) == 0:
            return [{"type": "info", "title": "No Numerical Features", "text": "No numeric variables found in this dataset."}]

        # Find highest variance / dispersion column (normalized by mean if mean != 0)
        coef_vars = {}
        for col in num_cols:
            clean = df[col].dropna().to_numpy()
            if len(clean) > 1:
                mean_val = np.mean(clean)
                std_val = np.std(clean)
                if abs(mean_val) > 1e-5:
                    coef_vars[col] = (std_val / abs(mean_val), mean_val, std_val, np.min(clean), np.max(clean))

        if coef_vars:
            highest_var_col = max(coef_vars.items(), key=lambda x: x[1][0])
            col_name, (cv, mean_v, std_v, min_v, max_v) = highest_var_col
            items.append({
                "type": "info",
                "title": f"Highest Relative Dispersion: {col_name}",
                "text": f"**{col_name}** exhibits the highest variability (Coefficient of Variation = {cv:.2f}), ranging from **{min_v:,.2f}** to **{max_v:,.2f}** with an average of **{mean_v:,.2f}** (std: {std_v:,.2f})."
            })

        # Check for skewness in numerical variables
        for col in num_cols[:4]:
            clean_s = df[col].dropna()
            if len(clean_s) > 3:
                skew = clean_s.skew()
                mean_v = clean_s.mean()
                med_v = clean_s.median()
                if skew > 1.0:
                    items.append({
                        "type": "info",
                        "title": f"Positive Skew in '{col}'",
                        "text": f"**{col}** is heavily right-skewed (skewness = {skew:.2f}) where the mean ({mean_v:,.2f}) is substantially higher than the median ({med_v:,.2f}), indicating a tail of high values."
                    })
                elif skew < -1.0:
                    items.append({
                        "type": "info",
                        "title": f"Negative Skew in '{col}'",
                        "text": f"**{col}** is heavily left-skewed (skewness = {skew:.2f}) with a cluster of values towards the upper range."
                    })

        return items

    @staticmethod
    def _correlation_insights(df: pd.DataFrame) -> List[Dict[str, str]]:
        """Identifies notable linear correlations and adds statistical context."""
        items = []
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) < 2:
            return [{"type": "info", "title": "Insufficient Numerical Columns", "text": "At least 2 numerical features are required for correlation analysis."}]

        corr_matrix = df[num_cols].corr()
        strong_pos = []
        strong_neg = []
        seen = set()

        for c1 in num_cols:
            for c2 in num_cols:
                if c1 != c2 and (c2, c1) not in seen:
                    val = corr_matrix.loc[c1, c2]
                    seen.add((c1, c2))
                    if not np.isnan(val):
                        if val >= 0.65:
                            strong_pos.append((c1, c2, val))
                        elif val <= -0.50:
                            strong_neg.append((c1, c2, val))

        if strong_pos:
            for c1, c2, val in strong_pos[:3]:
                items.append({
                    "type": "success",
                    "title": f"Strong Positive Association ({c1} ↔ {c2})",
                    "text": f"**{c1}** and **{c2}** have a strong positive correlation of **{val:.2f}**. As one increases, the other tends to increase proportionally."
                })

        if strong_neg:
            for c1, c2, val in strong_neg[:3]:
                items.append({
                    "type": "warning",
                    "title": f"Inverse Association ({c1} ↔ {c2})",
                    "text": f"**{c1}** and **{c2}** demonstrate an inverse negative correlation of **{val:.2f}**. Higher values in {c1} correspond with reduced values in {c2}."
                })

        if not strong_pos and not strong_neg:
            items.append({
                "type": "info",
                "title": "Moderate / Independent Numerical Features",
                "text": "Numerical variables in this dataset show largely independent or moderate relationships without extreme linear collinearity."
            })

        return items

    @staticmethod
    def _categorical_insights(df: pd.DataFrame) -> List[Dict[str, str]]:
        """Extracts observations on categorical class balances and dominant segments."""
        items = []
        cat_cols = df.select_dtypes(include=["object", "category"]).columns

        if len(cat_cols) == 0:
            return [{"type": "info", "title": "No Categorical Features", "text": "No text/category columns found in this dataset."}]

        for col in cat_cols[:4]:
            s = df[col].dropna()
            if len(s) > 0:
                counts = s.value_counts()
                top_name = counts.index[0]
                top_count = counts.iloc[0]
                top_pct = (top_count / len(s)) * 100
                unique_cnt = len(counts)

                if unique_cnt == 1:
                    items.append({
                        "type": "warning",
                        "title": f"Zero Variance in '{col}'",
                        "text": f"Column **{col}** has only 1 unique value ('{top_name}'). It provides no discriminating information."
                    })
                elif top_pct >= 50.0:
                    items.append({
                        "type": "info",
                        "title": f"Dominant Class in '{col}'",
                        "text": f"**'{top_name}'** dominates **{col}**, accounting for **{top_count:,} out of {len(s):,} entries ({top_pct:.1f}%)** across {unique_cnt} total unique categories."
                    })
                else:
                    items.append({
                        "type": "info",
                        "title": f"Category Distribution in '{col}'",
                        "text": f"**{col}** contains **{unique_cnt} categories**; the leading segment is **'{top_name}'** representing **{top_pct:.1f}%** of records."
                    })

        return items

    @staticmethod
    def _outlier_insights(df: pd.DataFrame) -> List[Dict[str, str]]:
        """Flags potential data anomalies detected through the IQR method."""
        items = []
        num_cols = df.select_dtypes(include=[np.number]).columns

        for col in num_cols:
            clean = df[col].dropna().to_numpy()
            if len(clean) >= 4:
                q1 = np.percentile(clean, 25)
                q3 = np.percentile(clean, 75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers = clean[(clean < lower) | (clean > upper)]
                if len(outliers) > 0:
                    pct = (len(outliers) / len(clean)) * 100
                    if pct >= 3.0:
                        items.append({
                            "type": "warning",
                            "title": f"Outliers in '{col}' ({pct:.1f}%)",
                            "text": f"Detected **{len(outliers)} statistical outlier(s)** in **{col}** outside the [{lower:.2f}, {upper:.2f}] IQR boundaries (Max: {np.max(outliers):.2f})."
                        })

        if not items:
            items.append({
                "type": "success",
                "title": "Clean Numerical Ranges",
                "text": "No significant statistical outlier clusters were detected across numerical features."
            })

        return items

    @staticmethod
    def _time_series_insights(df: pd.DataFrame) -> List[Dict[str, str]]:
        """Analyzes time bounds and temporal distribution if datetime columns are present."""
        items = []
        date_cols = df.select_dtypes(include=["datetime", "datetime64"]).columns.tolist()

        # Check object columns that look like dates
        if len(date_cols) == 0:
            for col in df.select_dtypes(include=["object"]).columns:
                if any(k in col.lower() for k in ["date", "time", "year"]):
                    try:
                        parsed = pd.to_datetime(df[col], errors="coerce")
                        if parsed.notna().sum() > len(df) * 0.7:
                            date_cols.append(col)
                    except Exception:
                        pass

        if not date_cols:
            return [{"type": "info", "title": "No Temporal Features", "text": "No datetime attributes detected in dataset."}]

        for col in date_cols[:2]:
            parsed = pd.to_datetime(df[col], errors="coerce").dropna()
            if len(parsed) > 0:
                min_d = parsed.min().strftime("%Y-%m-%d")
                max_d = parsed.max().strftime("%Y-%m-%d")
                span_days = (parsed.max() - parsed.min()).days
                items.append({
                    "type": "info",
                    "title": f"Temporal Span ({col})",
                    "text": f"Records in **{col}** span from **{min_d}** to **{max_d}** (a timeframe of **{span_days:,} days**)."
                })

        return items

    @staticmethod
    def generate_full_text_report(df: pd.DataFrame, dataset_name: str, cleaning_logs: Optional[List[str]] = None) -> str:
        """
        Generates a comprehensive, cleanly structured textual/markdown analysis report.
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        n_rows, n_cols = len(df), len(df.columns)
        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        cat_cols = list(df.select_dtypes(include=["object", "category", "bool"]).columns)
        missing_total = int(df.isna().sum().sum())
        dup_total = int(df.duplicated().sum())

        lines = [
            "=" * 70,
            "FILE DATA ANALYZER SYSTEM - COMPREHENSIVE DATA REPORT",
            "=" * 70,
            f"Generated On  : {now_str}",
            f"Dataset Name  : {dataset_name}",
            f"Total Records : {n_rows:,}",
            f"Total Columns : {n_cols}",
            f"Numerical Cols: {len(num_cols)} ({', '.join(num_cols) if num_cols else 'None'})",
            f"Categorical   : {len(cat_cols)} ({', '.join(cat_cols) if cat_cols else 'None'})",
            f"Missing Values: {missing_total:,}",
            f"Duplicates    : {dup_total:,}",
            "-" * 70,
            "",
            "1. DATA CLEANING & AUDIT LOG",
            "-" * 30,
        ]

        if cleaning_logs and len(cleaning_logs) > 0:
            for log_entry in cleaning_logs:
                lines.append(f"• {log_entry}")
        else:
            lines.append("• No data cleaning transformations applied yet (raw dataset state).")

        lines.extend([
            "",
            "2. NUMERICAL DESCRIPTIVE STATISTICS (NumPy & Pandas)",
            "-" * 50,
        ])

        if num_cols:
            for col in num_cols:
                clean = df[col].dropna().to_numpy()
                if len(clean) > 0:
                    lines.append(f"Column: {col}")
                    lines.append(f"  Count: {len(clean)} | Mean: {np.mean(clean):.4f} | Median: {np.median(clean):.4f} | Std: {np.std(clean, ddof=1) if len(clean)>1 else 0:.4f}")
                    lines.append(f"  Min: {np.min(clean):.4f} | Max: {np.max(clean):.4f} | Range: {np.ptp(clean):.4f} | Q1: {np.percentile(clean, 25):.4f} | Q3: {np.percentile(clean, 75):.4f}")
        else:
            lines.append("No numerical columns found.")

        lines.extend([
            "",
            "3. CATEGORICAL FREQUENCY BREAKDOWN",
            "-" * 40,
        ])

        if cat_cols:
            for col in cat_cols:
                top_v = df[col].mode()
                top_str = str(top_v.iloc[0]) if not top_v.empty else "N/A"
                lines.append(f"Column: {col} | Unique Categories: {df[col].nunique()} | Top Category: {top_str}")
        else:
            lines.append("No categorical columns found.")

        lines.extend([
            "",
            "4. AUTOMATIC RULE-BASED INSIGHTS",
            "-" * 35,
        ])

        all_insights = InsightsEngine.generate_all_insights(df)
        for category, cat_insights in all_insights.items():
            lines.append(f"[{category.upper()}]")
            for ins in cat_insights:
                clean_text = ins["text"].replace("**", "").replace("*", "")
                title = ins.get("title", "Note")
                lines.append(f"  * {title}: {clean_text}")

        lines.extend([
            "",
            "=" * 70,
            "End of Analysis Report - File Data Analyzer System",
            "=" * 70,
        ])

        return "\n".join(lines)

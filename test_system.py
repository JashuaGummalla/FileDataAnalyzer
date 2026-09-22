"""
Automated Verification Suite for File Data Analyzer System
Tests all modules: DataLoader, DataCleaner, StatisticalAnalyzer, Visualizer, and InsightsEngine.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add workspace directory to python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.data_loader import DataLoader
from modules.data_cleaner import DataCleaner
from modules.statistics import StatisticalAnalyzer
from modules.visualizations import Visualizer
from modules.insights import InsightsEngine


if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")


def run_all_tests():
    print("=" * 60)
    print("RUNNING AUTOMATED TEST SUITE FOR FILE DATA ANALYZER SYSTEM")
    print("=" * 60)

    # Test 1: Data Loader
    print("\n[TEST 1] Testing DataLoader with Sample Dataset...")
    df, err = DataLoader.load_sample_dataset()
    assert err is None, f"DataLoader error: {err}"
    assert df is not None, "DataFrame should not be None"
    assert len(df) > 0, "DataFrame should contain records"
    meta = DataLoader.get_dataset_metadata(df, "sample_sales_data.csv")
    assert meta["rows"] == len(df), "Row counts must match"
    print(f"  [PASS] DataLoader passed! Loaded {meta['rows']} rows, {meta['columns']} columns.")
    print(f"  [PASS] Metadata: Missing={meta['missing_values']}, Duplicates={meta['duplicate_rows']}")

    # Test 2: Data Cleaner (Duplicates)
    print("\n[TEST 2] Testing DataCleaner (Duplicate Removal)...")
    init_rows = len(df)
    df_dedup, removed = DataCleaner.remove_duplicates(df)
    assert removed == meta["duplicate_rows"], f"Expected {meta['duplicate_rows']} duplicates removed, got {removed}"
    assert len(df_dedup) == init_rows - removed
    print(f"  [PASS] Duplicate removal passed! Removed {removed} duplicates.")

    # Test 3: Data Cleaner (Missing Values Imputation)
    print("\n[TEST 3] Testing DataCleaner (Missing Values Imputation)...")
    df_imputed, summary = DataCleaner.handle_missing_values(df_dedup, strategy="fill_mean")
    assert "Sales" in df_imputed.columns
    print(f"  [PASS] Imputation passed! Imputed count: {summary['imputed_or_dropped_count']}")

    # Test 4: Data Cleaner (Header Standardization)
    print("\n[TEST 4] Testing DataCleaner (Header Standardization)...")
    df_clean_cols, mapping = DataCleaner.standardize_column_names(df, style="snake_case")
    for col in df_clean_cols.columns:
        assert " " not in col, f"Header '{col}' still contains spaces"
    print(f"  [PASS] Standardized headers: {list(df_clean_cols.columns)[:5]}")

    # Test 5: Statistical Analyzer (NumPy Math)
    print("\n[TEST 5] Testing StatisticalAnalyzer (NumPy Math)...")
    num_summary = StatisticalAnalyzer.compute_numerical_summary(df)
    assert not num_summary.empty, "Numerical summary should not be empty"
    assert "Sales" in num_summary.columns or "sales" in num_summary.columns
    sales_col = "Sales" if "Sales" in df.columns else "sales"
    np_inspect = StatisticalAnalyzer.get_column_numpy_inspection(df, sales_col)
    assert "np.mean()" in np_inspect
    assert "np.median()" in np_inspect
    assert "np.std()" in np_inspect
    print(f"  [PASS] NumPy calculations for '{sales_col}': Mean={np_inspect['np.mean()']:.2f}, Median={np_inspect['np.median()']:.2f}, Range={np_inspect['np.ptp() (Range)']:.2f}")

    # Test 6: IQR Outlier Detection
    print("\n[TEST 6] Testing IQR Outlier Detection...")
    outliers_df = StatisticalAnalyzer.detect_outliers_iqr(df)
    assert not outliers_df.empty, "Outlier table should not be empty"
    print(f"  [PASS] Outliers detected:\n{outliers_df[['Column', 'IQR', 'Outlier Count', 'Outlier %']]}")

    # Test 7: Data Quality Score
    print("\n[TEST 7] Testing Data Quality Score...")
    quality = StatisticalAnalyzer.calculate_data_quality_score(df)
    assert 0 <= quality["score"] <= 100
    print(f"  [PASS] Quality Score: {quality['score']}% ({quality['grade']})")

    # Test 8: Correlation Matrix
    print("\n[TEST 8] Testing Correlation Analysis...")
    corr_matrix, pairs = StatisticalAnalyzer.compute_correlation_matrix(df)
    assert not corr_matrix.empty, "Correlation matrix should not be empty"
    print(f"  [PASS] Correlation Matrix computed ({corr_matrix.shape[0]}x{corr_matrix.shape[1]}). Top pairs: {len(pairs)}")

    # Test 9: Visualizations (Matplotlib & Seaborn)
    print("\n[TEST 9] Testing Chart Generation (8 Chart Types)...")
    
    # 1. Hist
    fig1 = Visualizer.plot_histogram(df, column=sales_col)
    assert fig1 is not None
    plt.close(fig1)

    # 2. Bar
    fig2 = Visualizer.plot_bar_chart(df, x_col="Category")
    assert fig2 is not None
    plt.close(fig2)

    # 3. Line
    fig3 = Visualizer.plot_line_chart(df, x_col="Order_Date", y_col=sales_col)
    assert fig3 is not None
    plt.close(fig3)

    # 4. Scatter
    fig4 = Visualizer.plot_scatter(df, x_col="Quantity", y_col=sales_col, show_trendline=True)
    assert fig4 is not None
    plt.close(fig4)

    # 5. Box
    fig5 = Visualizer.plot_box_plot(df, y_col=sales_col, x_col="Category")
    assert fig5 is not None
    plt.close(fig5)

    # 6. Pie
    fig6, err = Visualizer.plot_pie_chart(df, column="Region")
    assert fig6 is not None
    plt.close(fig6)

    # 7. Count
    fig7 = Visualizer.plot_countplot(df, column="Customer_Type")
    assert fig7 is not None
    plt.close(fig7)

    # 8. Heatmap
    fig8 = Visualizer.plot_correlation_heatmap(corr_matrix)
    assert fig8 is not None
    plt.close(fig8)

    print("  [PASS] All 8 chart types generated successfully without errors!")

    # Test 10: Automatic Insights
    print("\n[TEST 10] Testing Insights Engine...")
    insights = InsightsEngine.generate_all_insights(df)
    assert "health" in insights and "numerical" in insights
    print(f"  [PASS] Generated {len(insights['health'])} health insights, {len(insights['numerical'])} numerical insights, {len(insights['correlation'])} correlation insights.")

    # Test 11: Full Text Report Generation
    print("\n[TEST 11] Testing Full Text Report Generator...")
    report = InsightsEngine.generate_full_text_report(df, "sample_sales_data.csv", ["Cleaned missing values"])
    assert len(report) > 200, "Report should be comprehensive"
    with open("outputs/reports/test_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  [PASS] Analysis report generated and saved ({len(report)} characters).")

    print("\n" + "=" * 60)
    print("[SUCCESS] ALL TESTS COMPLETED SUCCESSFULLY! ZERO FAILURES.")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

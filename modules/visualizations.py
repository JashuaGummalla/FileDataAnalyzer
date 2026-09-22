"""
Visualizations Module for File Data Analyzer System
Builds professional, modern publication-grade plots using Matplotlib and Seaborn.
Supports 8 chart types, customized themes, color palettes, and high-DPI export capabilities.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple

# Modern aesthetic configuration
plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"]
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.labelweight"] = "600"
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "700"
plt.rcParams["xtick.labelsize"] = 9.5
plt.rcParams["ytick.labelsize"] = 9.5
plt.rcParams["legend.fontsize"] = 9.5
plt.rcParams["figure.titlesize"] = 15


class Visualizer:
    """Generates various statistical charts using Matplotlib and Seaborn."""

    COLOR_PALETTES = {
        "Oceanic Sapphire (Default)": "mako",
        "Emerald Mint": "crest",
        "Sunset Crimson": "flare",
        "Viridis Luxe": "viridis",
        "Coolwarm Balance": "coolwarm",
        "Cyberpunk Violet": "rocket",
        "Deep Corporate": "deep",
        "Nordic Slate": "bone",
        "Vibrant Spectral": "Spectral",
        "Pastel Harmonious": "Set2",
    }

    @staticmethod
    def _create_figure(figsize=(9.5, 5.2), bg_color="#ffffff", ax_color="#f8fafc"):
        """Helper to create a high-DPI matplotlib figure and axes with modern borders."""
        fig, ax = plt.subplots(figsize=figsize, dpi=130)
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(ax_color)
        ax.grid(True, linestyle="--", alpha=0.45, color="#cbd5e1", zorder=0)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#94a3b8")
            ax.spines[spine].set_linewidth(1.0)
        return fig, ax

    @classmethod
    def plot_histogram(
        cls,
        df: pd.DataFrame,
        column: str,
        bins: int = 20,
        kde: bool = True,
        palette_name: str = "Oceanic Sapphire (Default)"
    ) -> plt.Figure:
        """Plots a Histogram with optional Kernel Density Estimate (KDE) and statistical markers."""
        fig, ax = cls._create_figure()
        data = df[column].dropna()
        
        palette_key = cls.COLOR_PALETTES.get(palette_name, "mako")
        try:
            color = sns.color_palette(palette_key)[2] if palette_key not in ["deep", "Set2"] else sns.color_palette(palette_key)[0]
        except Exception:
            color = "#0284c7"

        sns.histplot(
            data,
            bins=bins,
            kde=kde,
            color=color,
            ax=ax,
            edgecolor="#ffffff",
            linewidth=1.2,
            alpha=0.75,
            zorder=3
        )
        
        # Add statistical markers
        mean_val = np.mean(data)
        median_val = np.median(data)
        ax.axvline(mean_val, color="#ef4444", linestyle="--", linewidth=1.8, label=f"Mean: {mean_val:,.2f}", zorder=4)
        ax.axvline(median_val, color="#10b981", linestyle="-.", linewidth=1.8, label=f"Median: {median_val:,.2f}", zorder=4)

        ax.set_title(f"Distribution Analysis: {column}", pad=14, color="#0f172a")
        ax.set_xlabel(column, labelpad=8, color="#334155")
        ax.set_ylabel("Density / Frequency", labelpad=8, color="#334155")
        ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", framealpha=0.95, loc="upper right")
        plt.tight_layout()
        return fig

    @classmethod
    def plot_bar_chart(
        cls,
        df: pd.DataFrame,
        x_col: str,
        y_col: Optional[str] = None,
        agg_func: str = "sum",
        hue_col: Optional[str] = None,
        palette_name: str = "Oceanic Sapphire (Default)",
        top_n: int = 15
    ) -> plt.Figure:
        """Plots an aesthetic Bar Chart with clean value annotations."""
        fig, ax = cls._create_figure(figsize=(10, 5.5))
        palette_key = cls.COLOR_PALETTES.get(palette_name, "mako")

        if y_col and y_col in df.columns:
            agg_df = df.groupby(x_col)[y_col].agg(agg_func).reset_index()
            agg_df = agg_df.sort_values(by=y_col, ascending=False).head(top_n)
            
            sns.barplot(
                data=agg_df,
                x=x_col,
                y=y_col,
                hue=x_col,
                palette=palette_key,
                legend=False,
                ax=ax,
                edgecolor="#ffffff",
                linewidth=1.0,
                zorder=3
            )
            ax.set_ylabel(f"{agg_func.capitalize()} of {y_col}", color="#334155")
            ax.set_title(f"{agg_func.capitalize()} of {y_col} by {x_col} (Top {top_n})", pad=14, color="#0f172a")
            
            for p in ax.patches:
                height = p.get_height()
                if not np.isnan(height) and height != 0:
                    ax.annotate(
                        f"{height:,.1f}",
                        (p.get_x() + p.get_width() / 2.0, height),
                        ha="center",
                        va="bottom",
                        fontsize=8.5,
                        fontweight="bold",
                        color="#1e293b",
                        xytext=(0, 4),
                        textcoords="offset points"
                    )
        else:
            top_cats = df[x_col].value_counts().head(top_n).reset_index()
            top_cats.columns = [x_col, "Count"]
            sns.barplot(
                data=top_cats,
                x=x_col,
                y="Count",
                hue=x_col,
                palette=palette_key,
                legend=False,
                ax=ax,
                edgecolor="#ffffff",
                linewidth=1.0,
                zorder=3
            )
            ax.set_ylabel("Count", color="#334155")
            ax.set_title(f"Frequency Distribution: {x_col} (Top {top_n})", pad=14, color="#0f172a")
            
            for p in ax.patches:
                height = p.get_height()
                if not np.isnan(height):
                    ax.annotate(
                        f"{int(height):,}",
                        (p.get_x() + p.get_width() / 2.0, height),
                        ha="center",
                        va="bottom",
                        fontsize=8.5,
                        fontweight="bold",
                        color="#1e293b",
                        xytext=(0, 4),
                        textcoords="offset points"
                    )

        ax.set_xlabel(x_col, color="#334155")
        plt.xticks(rotation=30, ha="right", color="#334155")
        plt.yticks(color="#334155")
        plt.tight_layout()
        return fig

    @classmethod
    def plot_line_chart(
        cls,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        hue_col: Optional[str] = None,
        palette_name: str = "Oceanic Sapphire (Default)"
    ) -> plt.Figure:
        """Plots a Line Chart to show trends over time with glow-like markers."""
        fig, ax = cls._create_figure(figsize=(10, 5.2))
        palette_key = cls.COLOR_PALETTES.get(palette_name, "mako")

        plot_df = df.dropna(subset=[x_col, y_col]).sort_values(by=x_col)
        
        if hue_col and hue_col in df.columns:
            sns.lineplot(
                data=plot_df,
                x=x_col,
                y=y_col,
                hue=hue_col,
                marker="o",
                markersize=6,
                linewidth=2.2,
                palette=palette_key,
                ax=ax,
                zorder=3
            )
            ax.legend(title=hue_col, frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")
        else:
            try:
                color = sns.color_palette(palette_key)[1]
            except Exception:
                color = "#0284c7"
            sns.lineplot(
                data=plot_df,
                x=x_col,
                y=y_col,
                marker="o",
                markersize=6,
                color=color,
                linewidth=2.4,
                ax=ax,
                zorder=3
            )

        ax.set_title(f"Trend Analysis: {y_col} across {x_col}", pad=14, color="#0f172a")
        ax.set_xlabel(x_col, color="#334155")
        ax.set_ylabel(y_col, color="#334155")
        plt.xticks(rotation=30, ha="right", color="#334155")
        plt.yticks(color="#334155")
        plt.tight_layout()
        return fig

    @classmethod
    def plot_scatter(
        cls,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        hue_col: Optional[str] = None,
        show_trendline: bool = False,
        palette_name: str = "Oceanic Sapphire (Default)"
    ) -> plt.Figure:
        """Plots an elegant Scatter Plot with optional regression trendline."""
        fig, ax = cls._create_figure(figsize=(9.5, 5.5))
        palette_key = cls.COLOR_PALETTES.get(palette_name, "mako")
        plot_df = df.dropna(subset=[x_col, y_col])

        if show_trendline:
            sns.regplot(
                data=plot_df,
                x=x_col,
                y=y_col,
                ax=ax,
                scatter_kws={"alpha": 0.65, "color": "#0284c7", "s": 55},
                line_kws={"color": "#ef4444", "linewidth": 2.2, "label": "Linear Regression Trendline"},
                ci=95
            )
            ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")
        else:
            if hue_col and hue_col in df.columns:
                sns.scatterplot(
                    data=plot_df,
                    x=x_col,
                    y=y_col,
                    hue=hue_col,
                    palette=palette_key,
                    alpha=0.85,
                    s=65,
                    edgecolor="#ffffff",
                    linewidth=0.8,
                    ax=ax,
                    zorder=3
                )
                ax.legend(title=hue_col, frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")
            else:
                sns.scatterplot(
                    data=plot_df,
                    x=x_col,
                    y=y_col,
                    color="#0284c7",
                    alpha=0.85,
                    s=65,
                    edgecolor="#ffffff",
                    linewidth=0.8,
                    ax=ax,
                    zorder=3
                )

        corr = plot_df[x_col].corr(plot_df[y_col])
        corr_text = f"Pearson r = {corr:.3f}" if not np.isnan(corr) else "r = N/A"

        ax.set_title(f"Bivariate Correlation: {y_col} vs {x_col} ({corr_text})", pad=14, color="#0f172a")
        ax.set_xlabel(x_col, color="#334155")
        ax.set_ylabel(y_col, color="#334155")
        plt.tight_layout()
        return fig

    @classmethod
    def plot_box_plot(
        cls,
        df: pd.DataFrame,
        y_col: str,
        x_col: Optional[str] = None,
        hue_col: Optional[str] = None,
        palette_name: str = "Oceanic Sapphire (Default)"
    ) -> plt.Figure:
        """Plots a Box Plot showing quartiles, median, and outlier points."""
        fig, ax = cls._create_figure(figsize=(9.5, 5.5))
        palette_key = cls.COLOR_PALETTES.get(palette_name, "mako")

        flierprops = dict(marker="o", markerfacecolor="#ef4444", markeredgecolor="white", markersize=6, alpha=0.85)

        if x_col and x_col in df.columns:
            if hue_col and hue_col in df.columns:
                sns.boxplot(
                    data=df,
                    x=x_col,
                    y=y_col,
                    hue=hue_col,
                    palette=palette_key,
                    ax=ax,
                    flierprops=flierprops,
                    zorder=3
                )
                ax.legend(title=hue_col, frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")
            else:
                sns.boxplot(
                    data=df,
                    x=x_col,
                    y=y_col,
                    hue=x_col,
                    palette=palette_key,
                    legend=False,
                    ax=ax,
                    flierprops=flierprops,
                    zorder=3
                )
            ax.set_title(f"Box & Whisker Distribution: {y_col} Grouped by {x_col}", pad=14, color="#0f172a")
            plt.xticks(rotation=30, ha="right", color="#334155")
        else:
            sns.boxplot(
                data=df,
                y=y_col,
                color="#38bdf8",
                ax=ax,
                flierprops=flierprops,
                zorder=3
            )
            ax.set_title(f"Statistical Spread & Outliers: {y_col}", pad=14, color="#0f172a")

        ax.set_ylabel(y_col, color="#334155")
        plt.yticks(color="#334155")
        plt.tight_layout()
        return fig

    @classmethod
    def plot_pie_chart(
        cls,
        df: pd.DataFrame,
        column: str,
        max_slices: int = 7,
        donut: bool = True,
        palette_name: str = "Oceanic Sapphire (Default)"
    ) -> Tuple[Optional[plt.Figure], Optional[str]]:
        """Plots a Pie or Donut chart with modern clean wedge borders and slice legends."""
        counts = df[column].dropna().value_counts()
        if len(counts) == 0:
            return None, f"Column '{column}' contains no valid data."
        if len(counts) > 15:
            return None, f"Column '{column}' has {len(counts)} unique categories. Pie charts are not suitable for >15 categories. Use a Bar Chart instead."

        if len(counts) > max_slices:
            top_s = counts.head(max_slices)
            other_s = pd.Series({"Other": counts.iloc[max_slices:].sum()})
            counts = pd.concat([top_s, other_s])

        fig, ax = cls._create_figure(figsize=(8, 5.8))
        palette_key = cls.COLOR_PALETTES.get(palette_name, "mako")
        colors = sns.color_palette(palette_key, len(counts))

        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=counts.index,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            pctdistance=0.76 if donut else 0.6,
            wedgeprops={"edgecolor": "#ffffff", "linewidth": 2.2, "antialiased": True}
        )

        for t in texts:
            t.set_fontsize(9.5)
            t.set_color("#1e293b")
        for at in autotexts:
            at.set_fontsize(9)
            at.set_weight("bold")
            at.set_color("#0f172a")

        if donut:
            centre_circle = plt.Circle((0, 0), 0.56, fc="#ffffff", edgecolor="#e2e8f0", linewidth=1.5)
            ax.add_artist(centre_circle)
            ax.text(0, 0, f"{len(df[column].dropna()):,}\nTotal", ha="center", va="center", fontsize=11, fontweight="bold", color="#0f172a")

        ax.set_title(f"Proportional Share: {column}", pad=14, color="#0f172a")
        plt.tight_layout()
        return fig, None

    @classmethod
    def plot_countplot(
        cls,
        df: pd.DataFrame,
        column: str,
        hue_col: Optional[str] = None,
        palette_name: str = "Oceanic Sapphire (Default)"
    ) -> plt.Figure:
        """Plots a Seaborn Count Plot with frequency annotations on bars."""
        fig, ax = cls._create_figure(figsize=(9.5, 5.2))
        palette_key = cls.COLOR_PALETTES.get(palette_name, "mako")

        if hue_col and hue_col in df.columns:
            sns.countplot(
                data=df,
                x=column,
                hue=hue_col,
                palette=palette_key,
                ax=ax,
                edgecolor="#ffffff",
                linewidth=1.0,
                zorder=3
            )
            ax.legend(title=hue_col, frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")
        else:
            sns.countplot(
                data=df,
                x=column,
                hue=column,
                palette=palette_key,
                legend=False,
                ax=ax,
                edgecolor="#ffffff",
                linewidth=1.0,
                zorder=3
            )

        for p in ax.patches:
            height = p.get_height()
            if not np.isnan(height) and height > 0:
                ax.annotate(
                    f"{int(height):,}",
                    (p.get_x() + p.get_width() / 2.0, height),
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="bold",
                    color="#1e293b",
                    xytext=(0, 4),
                    textcoords="offset points"
                )

        ax.set_title(f"Category Frequency Count: {column}", pad=14, color="#0f172a")
        ax.set_xlabel(column, color="#334155")
        ax.set_ylabel("Occurrences", color="#334155")
        plt.xticks(rotation=30, ha="right", color="#334155")
        plt.yticks(color="#334155")
        plt.tight_layout()
        return fig

    @classmethod
    def plot_correlation_heatmap(
        cls,
        corr_matrix: pd.DataFrame,
        palette_name: str = "Coolwarm Balance"
    ) -> plt.Figure:
        """Plots an annotated correlation heatmap with clean colorbar and diverging colormap."""
        n_cols = len(corr_matrix.columns)
        figsize = (max(7.5, n_cols * 1.15), max(5.8, n_cols * 0.95))
        fig, ax = cls._create_figure(figsize=figsize)
        
        cmap = "coolwarm" if "Coolwarm" in palette_name else cls.COLOR_PALETTES.get(palette_name, "coolwarm")
        
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            vmin=-1.0,
            vmax=1.0,
            center=0,
            square=True,
            linewidths=1.5,
            linecolor="#ffffff",
            annot_kws={"fontsize": 9.5, "fontweight": "600"},
            cbar_kws={"shrink": 0.82, "label": "Pearson Correlation Coefficient (r)"},
            ax=ax
        )
        
        ax.set_title("Correlation Coefficient Matrix (Heatmap)", pad=14, color="#0f172a")
        plt.xticks(rotation=40, ha="right", color="#334155")
        plt.yticks(rotation=0, color="#334155")
        plt.tight_layout()
        return fig

    @classmethod
    def plot_missing_values(cls, df: pd.DataFrame) -> Tuple[Optional[plt.Figure], bool]:
        """Plots a horizontal bar chart of missing values across all columns."""
        missing = df.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if len(missing) == 0:
            return None, False

        fig, ax = cls._create_figure(figsize=(8.5, max(4.0, len(missing) * 0.55)))
        sns.barplot(
            x=missing.values,
            y=missing.index,
            hue=missing.index,
            palette="flare",
            legend=False,
            ax=ax,
            edgecolor="#ffffff",
            linewidth=1.0,
            zorder=3
        )

        for p in ax.patches:
            width = p.get_width()
            pct = (width / len(df)) * 100
            ax.annotate(
                f"{int(width):,} ({pct:.1f}%)",
                (width, p.get_y() + p.get_height() / 2.0),
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="#0f172a",
                xytext=(6, 0),
                textcoords="offset points"
            )

        ax.set_title("Missing Values Count by Attribute", pad=14, color="#0f172a")
        ax.set_xlabel("Missing Cells Count", color="#334155")
        ax.set_ylabel("Column Name", color="#334155")
        plt.tight_layout()
        return fig, True

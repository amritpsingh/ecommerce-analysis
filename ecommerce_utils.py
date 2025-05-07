#!/usr/bin/env python
# coding: utf-8

"""
E-commerce Analysis Utility Functions

This module contains common utility functions used across the e-commerce analysis notebooks:
- Data loading
- Display formatting
- Visualization functions

These functions provide consistent formatting and styling for the analysis outputs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display, HTML

# Set visualization style globally
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

def load_data(file_path='ecommerce_cleaned.parquet'):
    """
    Load the e-commerce dataset from a parquet file.

    Parameters:
        file_path (str): Path to the parquet file

    Returns:
        pd.DataFrame: The loaded DataFrame
    """
    try:
        df = pd.read_parquet(file_path, engine='pyarrow')
        print(f"Dataset loaded successfully with {df.shape[0]:,} rows and {df.shape[1]} columns.")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

def format_currency(value, precision=0):
    """Format a numeric value as currency with commas and dollar sign"""
    return f"${value:,.{precision}f}"

def format_percentage(value, precision=1):
    """Format a numeric value as percentage with specified precision"""
    return f"{value:.{precision}f}%"

def format_display_table(df, max_rows=10, precision=2):
    """
    Format a DataFrame as an HTML table with styling for better presentation.

    Parameters:
        df (pd.DataFrame): DataFrame to display
        max_rows (int): Maximum number of rows to display
        precision (int): Decimal precision for floating point numbers

    Returns:
        HTML display object
    """
    styled_df = df.head(max_rows).style.format(precision=precision, thousands=",")
    styled_df = styled_df.set_properties(**{'text-align': 'right'})
    styled_df = styled_df.set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center'), ('font-weight', 'bold')]},
        {'selector': '.row_heading, .blank', 'props': [('text-align', 'left')]}
    ])

    return HTML(styled_df.to_html())

def create_bar_chart(df, x, y, title, x_label, y_label, color=None, text=None, **kwargs):
    """
    Create a formatted and interactive Plotly bar chart.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data
        x (str): Column name for x-axis
        y (str): Column name for y-axis
        title (str): Chart title
        x_label (str): X-axis label
        y_label (str): Y-axis label
        color (str, optional): Column name for color encoding
        text (str or list, optional): Column name for bar labels or list of labels
        **kwargs: Additional parameters for px.bar

    Returns:
        plotly.graph_objects.Figure: The bar chart figure
    """
    # Check if y is a list (for grouped bars)
    if isinstance(y, list):
        # Use px.bar directly for multiple y values
        fig = px.bar(
            df, 
            x=x, 
            y=y,
            title=title,
            labels={x: x_label, 'value': y_label},
            barmode=kwargs.pop('barmode', 'group'),
            template='plotly_white',
            **kwargs
        )
    else:
        # Single y column
        fig = px.bar(
            df, 
            x=x, 
            y=y,
            color=color,
            title=title,
            labels={x: x_label, y: y_label},
            template='plotly_white',
            **kwargs
        )

        # Add text labels if provided
        if text is not None:
            if isinstance(text, str) and text in df.columns:
                fig.update_traces(text=df[text], textposition='outside')
            else:
                fig.update_traces(text=text, textposition='outside')

    # Improve layout
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        xaxis_tickangle=-45,
        margin=dict(l=40, r=40, t=60, b=80)
    )

    # Format axes for financial data
    if 'Profit' in y or 'Sales' in y or 'Revenue' in y:
        if not isinstance(y, list):  # Only apply for single y values
            fig.update_layout(yaxis_tickprefix='$', yaxis_tickformat=',')

    if 'Percentage' in y or 'Margin' in y:
        if not isinstance(y, list):  # Only apply for single y values
            fig.update_layout(yaxis_ticksuffix='%')

    return fig

def bar_chart(x, y, df, plot_title, x_label, y_label, percentage_col=None, scale='millions', 
              xtickangle=-45, **kwargs):
    """
    Create a formatted and interactive Plotly bar chart with customizable scaling and labels.

    Parameters:
        x (str): Column name for the x-axis.
        y (str): Column name for the y-axis (numeric values).
        df (pd.DataFrame): DataFrame containing the data.
        plot_title (str): Title for the chart.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        percentage_col (str, optional): Column name for percentage values. If provided, 
            the label will include the percentage value.
        scale (str): Scaling for value labels. Options:
            - 'none': No scaling (the value is shown as an integer).
            - 'thousands': Divides values by 1,000 and adds a "K" suffix.
            - 'millions': Divides values by 1,000,000 and adds an "M" suffix.
            Default is 'millions'.
        xtickangle (int): Angle for x-axis tick labels. Default is -45 degrees.
        **kwargs: Additional keyword arguments for px.bar (such as color, color_continuous_scale, etc.).

    Returns:
        plotly.graph_objects.Figure: The generated figure object
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_chart = df.copy()

    # Determine scale factor and suffix based on the scale parameter
    scale = scale.lower()
    if scale == 'thousands':
        factor = 1/1000
        suffix = 'K'
    elif scale == 'millions':
        factor = 1/1000000
        suffix = 'M'
    else:
        factor = 1
        suffix = ''

    # Create a new label column
    if percentage_col and percentage_col in df_chart.columns:
        # When scaling is used, round the scaled values to 1 decimal point
        if factor != 1:
            df_chart['label'] = df_chart.apply(
                lambda row: f'${row[y] * factor:,.1f}{suffix} ({row[percentage_col]:.1f}%)', axis=1)
        else:
            df_chart['label'] = df_chart.apply(
                lambda row: f'${row[y]:,.0f} ({row[percentage_col]:.1f}%)', axis=1)
    else:
        if factor != 1:
            df_chart['label'] = df_chart[y].apply(lambda v: f'${v * factor:,.1f}{suffix}')
        else:
            df_chart['label'] = df_chart[y].apply(lambda v: f'${v:,.0f}')

    # Create the Plotly Express bar chart
    fig = px.bar(
        df_chart,
        x=x,
        y=y,
        text='label',
        title=plot_title,
        labels={x: x_label, y: y_label},
        template='plotly_white',
        **kwargs
    )

    # Update traces to position labels outside the bars
    fig.for_each_trace(lambda trace: trace.update(texttemplate='%{text}', 
                                                textposition='outside', 
                                                cliponaxis=False))

    # Update layout settings
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        xaxis_tickangle=xtickangle,
        uniformtext_minsize=10,
        uniformtext_mode='hide'
    )

    return fig

def create_line_chart(df, x, y, title, x_label, y_label, color=None, markers=True, **kwargs):
    """
    Create a formatted and interactive Plotly line chart.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data
        x (str): Column name for x-axis
        y (str): Column name for y-axis
        title (str): Chart title
        x_label (str): X-axis label
        y_label (str): Y-axis label
        color (str, optional): Column name for color encoding
        markers (bool): Whether to show markers on the line
        **kwargs: Additional parameters for px.line

    Returns:
        plotly.graph_objects.Figure: The line chart figure
    """
    fig = px.line(
        df, 
        x=x, 
        y=y,
        color=color,
        title=title,
        labels={x: x_label, y: y_label},
        markers=markers,
        template='plotly_white',
        **kwargs
    )

    # Improve layout
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    # Format axes for financial data
    if 'Profit' in y or 'Sales' in y or 'Revenue' in y:
        fig.update_layout(yaxis_tickprefix='$', yaxis_tickformat=',')

    if 'Percentage' in y or 'Margin' in y:
        fig.update_layout(yaxis_ticksuffix='%')

    return fig

def line_chart(x, y, df, plot_title, x_label, y_label, text_col=None, markers=True, 
               highlight_max=False, highlight_min=False, **kwargs):
    """
    Create a formatted and interactive Plotly line chart with customizable options.

    Parameters:
        x (str): Column name for the x-axis
        y (str): Column name for the y-axis
        df (pd.DataFrame): DataFrame containing the data
        plot_title (str): Title for the chart
        x_label (str): Label for the x-axis
        y_label (str): Label for the y-axis
        text_col (str, optional): Column name for point labels
        markers (bool): Whether to show markers on the line
        highlight_max (bool): Whether to highlight the maximum value point
        highlight_min (bool): Whether to highlight the minimum value point
        **kwargs: Additional keyword arguments for px.line

    Returns:
        plotly.graph_objects.Figure: The generated figure object
    """
    # Create a copy of the dataframe to avoid modifying the original
    df_chart = df.copy()

    # Create the line chart
    fig = px.line(
        df_chart,
        x=x,
        y=y,
        markers=markers,
        title=plot_title,
        labels={x: x_label, y: y_label},
        template='plotly_white',
        **kwargs
    )

    # Add text labels if provided
    if text_col and text_col in df_chart.columns:
        fig.update_traces(text=df_chart[text_col], textposition='top center')

    # Update line and marker appearance
    fig.update_traces(line=dict(width=2), marker=dict(size=8))

    # Highlight maximum value if requested
    if highlight_max:
        max_idx = df_chart[y].idxmax()
        max_point = df_chart.loc[max_idx]

        fig.add_scatter(
            x=[max_point[x]],
            y=[max_point[y]],
            mode='markers',
            marker=dict(color='green', size=15, symbol='circle'),
            name='Highest Value',
            hoverinfo='y'
        )

    # Highlight minimum value if requested
    if highlight_min:
        min_idx = df_chart[y].idxmin()
        min_point = df_chart.loc[min_idx]

        fig.add_scatter(
            x=[min_point[x]],
            y=[min_point[y]],
            mode='markers',
            marker=dict(color='red', size=15, symbol='circle'),
            name='Lowest Value',
            hoverinfo='y'
        )

    return fig

def create_scatter_plot(df, x, y, title, x_label, y_label, color=None, size=None, hover_name=None, **kwargs):
    """
    Create a formatted and interactive Plotly scatter plot.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data
        x (str): Column name for x-axis
        y (str): Column name for y-axis
        title (str): Chart title
        x_label (str): X-axis label
        y_label (str): Y-axis label
        color (str, optional): Column name for color encoding
        size (str, optional): Column name for size encoding
        hover_name (str, optional): Column name for hover labels
        **kwargs: Additional parameters for px.scatter

    Returns:
        plotly.graph_objects.Figure: The scatter plot figure
    """
    fig = px.scatter(
        df, 
        x=x, 
        y=y,
        color=color,
        size=size,
        hover_name=hover_name,
        title=title,
        labels={x: x_label, y: y_label},
        template='plotly_white',
        **kwargs
    )

    # Improve layout
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    # Format axes for financial data
    if 'Profit' in x or 'Sales' in x or 'Revenue' in x:
        fig.update_layout(xaxis_tickprefix='$', xaxis_tickformat=',')

    if 'Profit' in y or 'Sales' in y or 'Revenue' in y:
        fig.update_layout(yaxis_tickprefix='$', yaxis_tickformat=',')

    if 'Percentage' in x or 'Margin' in x:
        fig.update_layout(xaxis_ticksuffix='%')

    if 'Percentage' in y or 'Margin' in y:
        fig.update_layout(yaxis_ticksuffix='%')

    return fig

def create_pie_chart(df, values, names, title, **kwargs):
    """
    Create a formatted and interactive Plotly pie chart.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data
        values (str): Column name for slice values
        names (str): Column name for slice names
        title (str): Chart title
        **kwargs: Additional parameters for px.pie

    Returns:
        plotly.graph_objects.Figure: The pie chart figure
    """
    fig = px.pie(
        df,
        values=values,
        names=names,
        title=title,
        template='plotly_white',
        **kwargs
    )

    # Improve layout
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hole=kwargs.get('hole', 0.4)
    )

    return fig

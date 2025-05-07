#!/usr/bin/env python
# coding: utf-8

# # E-commerce Customer Segmentation Analysis
# ### Executive Summary
# This analysis examines customer behaviors and segmentation within our e-commerce platform to understand who's driving our business performance. Our findings reveal:
# 
# 73.9% of customers are one-time purchasers yet generate 55.9% of revenue
# "Champions" (our high-value customers) represent 22.9% of customers but drive 43.1% of revenue
# Gender differences are minimal with males (54.7% of customers) showing similar behaviors to females
# Web platform dominates with 93.4% of revenue vs. mobile (6.6%)
# Fashion category preference varies by segment - Champions and Potential Loyalists strongly prefer Fashion (58-60%) while At-Risk customers favor Home & Furniture (56.9%)
# 
# The insights in this notebook will help optimize marketing strategies, personalization efforts, and customer retention initiatives to maximize lifetime value across different segments.
# ### Introduction
# Understanding who our customers are and how they behave is crucial for effective e-commerce strategy. This analysis aims to:
# 
# Identify key customer segments based on demographics and purchasing patterns
# Analyze how different segments contribute to revenue and profitability
# Examine product category preferences across customer segments
# Understand discount sensitivity and device preferences by segment
# Provide actionable recommendations for targeting and retention
# 
# ## Setup & Data Preparation

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, HTML

# Import our custom utility functions
from ecommerce_utils import (load_data, format_currency, format_percentage, 
                           format_display_table, create_bar_chart, 
                           create_line_chart, create_scatter_plot, create_pie_chart)

# Load the data
df = load_data('ecommerce_cleaned.parquet')


# ## 1. Demographic Customer Segmentation
# ### 1.1 Gender Analysis

# In[2]:


def analyze_gender_demographics(df):
    """
    Analyze customer demographics and purchasing behavior by gender.

    Parameters:
        df (pd.DataFrame): The e-commerce dataset

    Returns:
        tuple: (gender_metrics DataFrame, bar chart figure)
    """
    # Calculate metrics by gender
    gender_metrics = df.groupby('Gender', observed=True).agg({
        'Customer_Id': 'nunique',      # Unique customers
        'Order_Id': 'nunique',         # Number of orders
        'Total_Sales': 'sum',          # Total revenue
        'Total_Profit': 'sum',         # Total profit
        'Discount': 'mean'             # Average discount rate
    }).reset_index()

    # Calculate derived metrics
    gender_metrics['Orders_Per_Customer'] = (gender_metrics['Order_Id'] / gender_metrics['Customer_Id']).round(2)
    gender_metrics['Avg_Order_Value'] = (gender_metrics['Total_Sales'] / gender_metrics['Order_Id']).round(2)
    gender_metrics['Profit_Margin'] = (gender_metrics['Total_Profit'] / gender_metrics['Total_Sales'] * 100).round(1)
    gender_metrics['Avg_Discount'] = (gender_metrics['Discount'] * 100).round(1)
    gender_metrics['Customer_Distribution'] = (gender_metrics['Customer_Id'] / gender_metrics['Customer_Id'].sum() * 100).round(1)

    # Create metrics for visualization
    metrics_to_viz = gender_metrics[['Gender', 'Orders_Per_Customer', 'Avg_Order_Value', 'Profit_Margin']]

    # Create bar chart
    bar_fig = create_bar_chart(
        metrics_to_viz,
        x='Gender',
        y=['Orders_Per_Customer', 'Avg_Order_Value', 'Profit_Margin'],
        title='Key Performance Metrics by Gender',
        x_label='Gender',
        y_label='Value',
        barmode='group'
    )

    return gender_metrics, bar_fig

# Run the analysis
gender_data, gender_chart = analyze_gender_demographics(df)

# Display results
print("Customer Demographics by Gender:")
display(format_display_table(gender_data))

# Show chart
gender_chart.show()


# **Key Insights on Gender Demographics**
# Our gender demographic analysis reveals remarkably balanced purchasing behaviors between male and female customers:
# 
# 1. **Customer Distribution:** Males constitute 54.7% of our customer base, while females make up 45.3%.
# 2. **Purchase Frequency:** Both genders show nearly identical purchase frequency - males average 1.17 orders per customer compared to 1.16 for females.
# 3. **Order Value:** Female customers have slightly higher average order values at \$390.50 versus \$376.55 for males, a difference of approximately 3.7%.
# 4. **Profit Margins:** We maintain slightly higher profit margins on female customers (44.0%) compared to males (43.4%).
# 5. **Discount Usage:** Male customers tend to receive higher average discounts (31.4%) compared to females (29.2%), a difference that may explain the slight margin difference.
# 
# These findings suggest our product assortment and pricing strategy appeal similarly to both genders. The minimal differences in purchasing behavior indicate that gender-specific marketing might not yield significant returns compared to other segmentation approaches.
# 
# ### 1.2 Device Type Analysis

# In[3]:


def analyze_device_usage(df):
    """
    Analyze customer device usage patterns and their impact on purchasing behavior.

    Parameters:
        df (pd.DataFrame): The e-commerce dataset

    Returns:
        tuple: (device_metrics DataFrame, bar chart figure)
    """
    # Calculate metrics by device type
    device_metrics = df.groupby('Device_Type', observed=True).agg({
        'Customer_Id': 'nunique',      # Unique customers
        'Order_Id': 'nunique',         # Number of orders
        'Total_Sales': 'sum',          # Total revenue
        'Total_Profit': 'sum',         # Total profit
        'Discount': 'mean'             # Average discount rate
    }).reset_index()

    # Calculate derived metrics
    device_metrics['Orders_Per_Customer'] = (device_metrics['Order_Id'] / device_metrics['Customer_Id']).round(2)
    device_metrics['Avg_Order_Value'] = (device_metrics['Total_Sales'] / device_metrics['Order_Id']).round(2)
    device_metrics['Profit_Margin'] = (device_metrics['Total_Profit'] / device_metrics['Total_Sales'] * 100).round(1)
    device_metrics['Avg_Discount'] = (device_metrics['Discount'] * 100).round(1)
    device_metrics['Revenue_Share'] = (device_metrics['Total_Sales'] / device_metrics['Total_Sales'].sum() * 100).round(1)

    # Create metrics for visualization
    metrics_to_viz = device_metrics[['Device_Type', 'Orders_Per_Customer', 'Avg_Order_Value', 'Profit_Margin', 'Revenue_Share']]

    # Create paired bar and pie charts
    bar_fig = create_bar_chart(
        metrics_to_viz,
        x='Device_Type',
        y=['Orders_Per_Customer', 'Avg_Order_Value', 'Profit_Margin'],
        title='Key Metrics by Device Type',
        x_label='Device Type',
        y_label='Value',
        barmode='group'
    )

    pie_fig = create_pie_chart(
        device_metrics,
        values='Total_Sales',
        names='Device_Type',
        title='Revenue Share by Device Type',
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    return device_metrics, bar_fig, pie_fig

# Run the analysis
device_data, device_bar_chart, device_pie_chart = analyze_device_usage(df)

# Display results
print("Device Usage Analysis:")
display(format_display_table(device_data))

# Show charts
device_bar_chart.show()
device_pie_chart.show()


# **Key Insights on Device Usage**
# Our device usage analysis reveals a significant disparity between web and mobile platforms:
# 
# 1. **Web Dominance:** Web platform accounts for an overwhelming 93.4% of total revenue, making it our primary sales channel.
# 2. **Higher Web Engagement:** Web users show substantially higher engagement with 1.29 orders per customer compared to just 1.03 for mobile users, suggesting the web experience drives repeat purchases more effectively.
# 3. **Higher Web Order Value:** Web orders average \$384.96, approximately 8.3% higher than mobile orders at \$355.35.
# 4. **Consistent Profit Margins:** Despite differences in usage and order value, profit margins remain remarkably similar between web (43.7%) and mobile (43.8%).
# 5. **Small Mobile Customer Base:** Only 3,557 customers (8.8% of total) use our mobile platform, indicating a significant opportunity for mobile customer acquisition.
# 
# These insights suggest our web platform is well-optimized while our mobile experience may need improvement to drive both customer acquisition and engagement. The similar profit margins indicate we're maintaining consistent pricing and discount strategies across both platforms.
# 
# ### 1.3 Customer Login Type Analysis

# In[4]:


def analyze_login_types(df):
    """
    Analyze customer behavior across different login types.

    Parameters:
        df (pd.DataFrame): The e-commerce dataset

    Returns:
        tuple: (login_metrics DataFrame, bar chart figure)
    """
    # Calculate metrics by login type
    login_metrics = df.groupby('Customer_Login_type', observed=True).agg({
        'Customer_Id': 'nunique',      # Unique customers
        'Order_Id': 'nunique',         # Number of orders
        'Total_Sales': 'sum',          # Total revenue
        'Total_Profit': 'sum',         # Total profit
        'Discount': 'mean'             # Average discount rate
    }).reset_index()

    # Calculate derived metrics
    login_metrics['Orders_Per_Customer'] = (login_metrics['Order_Id'] / login_metrics['Customer_Id']).round(2)
    login_metrics['Avg_Order_Value'] = (login_metrics['Total_Sales'] / login_metrics['Order_Id']).round(2)
    login_metrics['Profit_Margin'] = (login_metrics['Total_Profit'] / login_metrics['Total_Sales'] * 100).round(1)
    login_metrics['Avg_Discount'] = (login_metrics['Discount'] * 100).round(1)
    login_metrics['Customer_Distribution'] = (login_metrics['Customer_Id'] / login_metrics['Customer_Id'].sum() * 100).round(1)
    login_metrics['Revenue_Percentage'] = (login_metrics['Total_Sales'] / login_metrics['Total_Sales'].sum() * 100).round(1)

    # Create comparison visualization
    fig = create_bar_chart(
        login_metrics,
        x='Customer_Login_type',
        y=['Avg_Order_Value', 'Profit_Margin'],
        title='Order Value and Margin by Customer Login Type',
        x_label='Login Type',
        y_label='Value',
        barmode='group'
    )

    # Create pie chart for distribution
    pie_fig = create_pie_chart(
        login_metrics,
        values='Customer_Id',
        names='Customer_Login_type',
        title='Customer Distribution by Login Type',
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    return login_metrics, fig, pie_fig

# Run the analysis
login_data, login_chart, login_pie_chart = analyze_login_types(df)

# Display results
print("Customer Login Type Analysis:")
display(format_display_table(login_data))

# Show charts
login_chart.show()
login_pie_chart.show()


# **Key Insights on Customer Login Types**
# Our login type analysis reveals important patterns in customer account usage:
# 
# 1. **Member Dominance:** Registered members constitute 94.6% of our customer base and generate 95.7% of total revenue, highlighting the importance of our membership program.
# 2. **Higher Member Engagement:** Members show significantly higher engagement with 1.30 orders per customer compared to 1.01 for guests and 1.00 for new customers.
# 3. **New Customer Value:** "New" login type customers have the highest average order value at \$571.04, substantially higher than members (\$382.81) and guests (\$386.02), though they represent only 0.1% of customers.
# 4. **Consistent Profitability:** Profit margins remain relatively consistent across login types, ranging from 42.6% for First Signup to 44.8% for New customers.
# 5. **Member-Focused Strategy:** The overwhelming dominance of members suggests our business model effectively incentivizes account creation and login.
# 
# These findings indicate that converting guests to members should remain a priority, while the high average order value of new customers suggests potential for targeted first-purchase promotions. The strong member engagement demonstrates successful retention strategies, which should be maintained and enhanced.
# 
# ## 2. Purchase Behavior Segmentation

# In[5]:


def create_customer_level_metrics(df):
    """
    Aggregate transaction data to create customer-level metrics for segmentation.

    Parameters:
        df (pd.DataFrame): The e-commerce dataset

    Returns:
        pd.DataFrame: Customer-level metrics
    """
    # Create customer-level aggregated metrics
    customer_metrics = df.groupby('Customer_Id', observed=True).agg({
        'Order_Id': 'nunique',             # Number of orders (frequency)
        'Total_Sales': 'sum',              # Total spend (monetary)
        'Order_Date': 'max',               # Date of last purchase (recency)
        'Product_Category': lambda x: len(x.unique()),  # Number of categories purchased
        'Product': lambda x: len(x.unique()),  # Number of unique products purchased
        'Discount': 'mean',                # Average discount rate
        'Profit_Margin': 'mean'            # Average profit margin
    }).reset_index()

    # Rename columns for clarity
    customer_metrics.columns = ['Customer_Id', 'Purchase_Frequency', 'Total_Spend', 
                               'Last_Purchase_Date', 'Categories_Purchased', 
                               'Products_Purchased', 'Avg_Discount', 'Avg_Profit_Margin']

    # Calculate days since last purchase (recency)
    latest_date = df['Order_Date'].max()
    customer_metrics['Days_Since_Purchase'] = (latest_date - customer_metrics['Last_Purchase_Date']).dt.days

    # Calculate average order value
    customer_metrics['Avg_Order_Value'] = (customer_metrics['Total_Spend'] / customer_metrics['Purchase_Frequency']).round(2)

    return customer_metrics

def segment_customers_by_behavior(customer_metrics):
    """
    Create customer segments based on purchase frequency, monetary value, and recency.

    Parameters:
        customer_metrics (pd.DataFrame): Customer-level metrics

    Returns:
        pd.DataFrame: Customer metrics with segment labels
    """
    # Show distribution of purchase frequency
    print("Purchase Frequency Distribution:")
    print(customer_metrics['Purchase_Frequency'].value_counts().sort_index().head(10))

    # Customer Frequency Segmentation - use 'duplicates=drop' to handle duplicate bin edges
    try:
        customer_metrics['Frequency_Segment'] = pd.qcut(
            customer_metrics['Purchase_Frequency'],
            q=[0, 0.25, 0.5, 0.75, 1],
            labels=['One-time', 'Occasional', 'Regular', 'Frequent'],
            duplicates='drop'  # Handle duplicate edges
        )
    except ValueError:
        # Alternative approach: use bins instead of quantiles if we still have issues
        freq_bins = [0, 1, 2, 4, customer_metrics['Purchase_Frequency'].max()]
        customer_metrics['Frequency_Segment'] = pd.cut(
            customer_metrics['Purchase_Frequency'],
            bins=freq_bins,
            labels=['One-time', 'Occasional', 'Regular', 'Frequent'],
            include_lowest=True
        )

    # Customer Monetary Segmentation
    try:
        customer_metrics['Monetary_Segment'] = pd.qcut(
            customer_metrics['Total_Spend'],
            q=[0, 0.25, 0.5, 0.75, 1],
            labels=['Low', 'Medium', 'High', 'Top'],
            duplicates='drop'
        )
    except ValueError:
        # If we still encounter issues, use rank to break ties
        customer_metrics['Monetary_Segment'] = pd.qcut(
            customer_metrics['Total_Spend'].rank(method='first'),
            q=4,
            labels=['Low', 'Medium', 'High', 'Top']
        )

    # Customer Recency Segmentation (lower days = more recent)
    try:
        customer_metrics['Recency_Segment'] = pd.qcut(
            customer_metrics['Days_Since_Purchase'],
            q=[0, 0.25, 0.5, 0.75, 1],
            labels=['Recent', 'Active', 'Lapsed', 'Inactive'],
            duplicates='drop'
        )
    except ValueError:
        # Alternative approach using rank method
        customer_metrics['Recency_Segment'] = pd.qcut(
            customer_metrics['Days_Since_Purchase'].rank(method='first'),
            q=4,
            labels=['Recent', 'Active', 'Lapsed', 'Inactive']
        )

    # Print summary of customer segments
    print("\nCustomer Segmentation Summary:")
    print(f"Total unique customers: {len(customer_metrics):,}")

    print("\nFrequency Segments:")
    print(customer_metrics['Frequency_Segment'].value_counts())

    print("\nMonetary Segments:")
    print(customer_metrics['Monetary_Segment'].value_counts())

    print("\nRecency Segments:")
    print(customer_metrics['Recency_Segment'].value_counts())

    return customer_metrics

def analyze_frequency_segments(customer_metrics):
    """
    Analyze metrics across frequency segments.

    Parameters:
        customer_metrics (pd.DataFrame): Customer metrics with segment labels

    Returns:
        tuple: (frequency_segment_metrics DataFrame, pie chart figure)
    """
    # Calculate metrics by frequency segment
    frequency_segment_metrics = customer_metrics.groupby('Frequency_Segment').agg({
        'Customer_Id': 'count',
        'Total_Spend': 'sum',
        'Avg_Order_Value': 'mean',
        'Avg_Profit_Margin': 'mean',
        'Categories_Purchased': 'mean'
    }).reset_index()

    # Calculate percentage metrics
    frequency_segment_metrics['Customers_Percentage'] = (frequency_segment_metrics['Customer_Id'] / 
                                                      frequency_segment_metrics['Customer_Id'].sum() * 100).round(1)
    frequency_segment_metrics['Revenue_Percentage'] = (frequency_segment_metrics['Total_Spend'] / 
                                                    frequency_segment_metrics['Total_Spend'].sum() * 100).round(1)

    # Create pie chart for revenue contribution
    pie_fig = create_pie_chart(
        frequency_segment_metrics,
        values='Revenue_Percentage',
        names='Frequency_Segment',
        title='Revenue Contribution by Purchase Frequency Segment',
        color_discrete_sequence=px.colors.sequential.Viridis
    )

    # Create bar chart comparing customer % vs revenue %
    bar_fig = create_bar_chart(
        frequency_segment_metrics,
        x='Frequency_Segment',
        y=['Customers_Percentage', 'Revenue_Percentage'],
        title='Customer Distribution vs. Revenue Contribution by Frequency',
        x_label='Purchase Frequency Segment',
        y_label='Percentage (%)',
        barmode='group'
    )

    return frequency_segment_metrics, pie_fig, bar_fig

# Run the analysis
customer_metrics = create_customer_level_metrics(df)
segmented_customers = segment_customers_by_behavior(customer_metrics)
frequency_data, frequency_pie, frequency_bar = analyze_frequency_segments(segmented_customers)

# Display frequency segment results
print("\nMetrics by Purchase Frequency Segment:")
display(format_display_table(frequency_data))

# Show charts
frequency_pie.show()
frequency_bar.show()


# **Key Insights on Purchase Behavior Segments**
# Our purchase behavior segmentation reveals striking patterns in customer purchasing habits:
# 
# 1. **One-Time Customer Dominance:** One-time purchasers represent 73.9% of our customer base but contribute only 55.9% of revenue, highlighting a significant opportunity for conversion to repeat customers.
# 2. **Value of Repeat Customers:** Despite making up just 26.1% of customers, repeat buyers (Occasional, Regular, and Frequent combined) generate 44.1% of revenue, demonstrating their disproportionate value.
# 3. **Category Exploration:** As purchase frequency increases, so does the average number of categories purchased - from 1.00 for one-time customers to 1.94 for frequent buyers, showing how repeated engagement broadens purchasing scope.
# 4. **Order Value Progression:** Average order value increases with purchase frequency, from $380.99 for one-time customers to $393.73 for frequent customers, suggesting higher customer comfort and trust with repeated purchases.
# 5. **Profit Consistency:** Profit margins remain relatively stable across frequency segments (41.13% to 42.85%), indicating we maintain pricing discipline across customer groups.
# 
# These insights highlight the critical importance of converting one-time buyers into repeat customers, as each increase in purchase frequency correlates with higher average order values and broader category exploration, driving overall customer lifetime value.
# 
# ## 3. RFM (Recency, Frequency, Monetary) Analysis

# In[6]:


def perform_rfm_analysis(customer_metrics):
    """
    Perform RFM (Recency, Frequency, Monetary) analysis and create customer segments.

    Parameters:
        customer_metrics (pd.DataFrame): Customer-level metrics

    Returns:
        pd.DataFrame: Customer metrics with RFM segments
    """
    # Create RFM scores
    ## For Recency: lower is better (more recent purchase)
    customer_metrics['R_Score'] = pd.qcut(customer_metrics['Days_Since_Purchase'], 4, labels=[4, 3, 2, 1])
    ## For Frequency: higher is better (more purchases)
    customer_metrics['F_Score'] = pd.qcut(customer_metrics['Purchase_Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
    ## For Monetary: higher is better (more spending)
    customer_metrics['M_Score'] = pd.qcut(customer_metrics['Total_Spend'].rank(method='first'), 4, labels=[1, 2, 3, 4])

    # Convert to numeric for calculation
    customer_metrics['R_Score'] = pd.to_numeric(customer_metrics['R_Score'])
    customer_metrics['F_Score'] = pd.to_numeric(customer_metrics['F_Score'])
    customer_metrics['M_Score'] = pd.to_numeric(customer_metrics['M_Score'])

    # Calculate RFM Score
    customer_metrics['RFM_Score'] = customer_metrics['R_Score'] + customer_metrics['F_Score'] + customer_metrics['M_Score']

    # Create RFM Segments
    def create_rfm_segment(row):
        if row['RFM_Score'] >= 10:
            return 'Champions'
        elif (row['RFM_Score'] >= 8) and (row['R_Score'] >= 3):
            return 'Loyal Customers'
        elif (row['RFM_Score'] >= 8) and (row['R_Score'] < 3):
            return 'Potential Loyalists'
        elif (row['F_Score'] >= 3) and (row['R_Score'] < 3):
            return 'At Risk'
        elif (row['M_Score'] >= 3) and (row['F_Score'] < 3):
            return 'Big Spenders'
        elif (row['R_Score'] >= 3) and (row['F_Score'] < 3) and (row['M_Score'] < 3):
            return 'New Customers'
        elif (row['RFM_Score'] >= 5) and (row['RFM_Score'] < 8):
            return 'Need Attention'
        else:
            return 'Hibernating'

    customer_metrics['RFM_Segment'] = customer_metrics.apply(create_rfm_segment, axis=1)

    return customer_metrics

def analyze_rfm_segments(rfm_customer_metrics):
    """
    Analyze metrics across RFM segments.

    Parameters:
        rfm_customer_metrics (pd.DataFrame): Customer metrics with RFM segments

    Returns:
        tuple: (rfm_segment_metrics DataFrame, bar chart figure)
    """
    # Calculate metrics by RFM segment
    rfm_segment_metrics = rfm_customer_metrics.groupby('RFM_Segment').agg({
        'Customer_Id': 'count',
        'Total_Spend': 'sum',
        'Avg_Order_Value': 'mean',
        'Purchase_Frequency': 'mean',
        'Avg_Profit_Margin': 'mean'
    }).reset_index()

    # Calculate percentage metrics
    rfm_segment_metrics['Customer_Percentage'] = (rfm_segment_metrics['Customer_Id'] / 
                                                rfm_segment_metrics['Customer_Id'].sum() * 100).round(1)
    rfm_segment_metrics['Revenue_Percentage'] = (rfm_segment_metrics['Total_Spend'] / 
                                              rfm_segment_metrics['Total_Spend'].sum() * 100).round(1)

    # Sort by revenue percentage
    rfm_segment_metrics = rfm_segment_metrics.sort_values('Revenue_Percentage', ascending=False)

    # Create visualization
    bar_fig = create_bar_chart(
        rfm_segment_metrics,
        x='RFM_Segment',
        y=['Customer_Percentage', 'Revenue_Percentage'],
        title='Customer Distribution vs. Revenue Contribution by RFM Segment',
        x_label='Customer Segment',
        y_label='Percentage (%)',
        barmode='group'
    )

    # Create pie chart for revenue contribution
    pie_fig = create_pie_chart(
        rfm_segment_metrics,
        values='Revenue_Percentage',
        names='RFM_Segment',
        title='Revenue Contribution by RFM Segment',
        color_discrete_sequence=px.colors.qualitative.Bold
    )

    return rfm_segment_metrics, bar_fig, pie_fig

# Run the analysis
rfm_customers = perform_rfm_analysis(customer_metrics)
rfm_data, rfm_bar, rfm_pie = analyze_rfm_segments(rfm_customers)

# Display results
print("\nRFM Customer Segmentation:")
display(format_display_table(rfm_data))

# Show charts
rfm_bar.show()
rfm_pie.show()


# **Key Insights on RFM Segmentation**
# Our RFM analysis reveals powerful insights about customer value segments and their contribution to business performance:
# 
# 1. **Champions Drive Revenue:** "Champions" segment (22.9% of customers) contributes a disproportionate 43.1% of total revenue, making them our most valuable customer group with the highest purchase frequency (2.07) and strong average order value (\$494.06).
# 2. **Big Spenders Show Potential:** "Big Spenders" (11.0% of customers) have the highest average order value (\$655.19) but only purchase once on average, representing a significant opportunity for increasing purchase frequency.
# 3. **Loyalty Drives Value:** "Loyal Customers" (15.4%) and "Potential Loyalists" (8.6%) together contribute 26.5% of revenue, highlighting the cumulative value of customer retention.
# 4. **At-Risk Customers Need Attention:** The "At-Risk" segment (10.7% of customers) shows low average order value (\$237.62) and contributes only 5.3% of revenue, suggesting these customers may be losing engagement.
# 5. **Hibernating Customers Underperform:** "Hibernating" customers (9.5%) have the lowest average order value (\$150.44) and profit margins (39.71%), contributing just 2.9% of revenue.
# 
# These findings demonstrate the importance of tailoring marketing strategies to each segment: retention programs for Champions, frequency-boosting campaigns for Big Spenders, re-engagement efforts for At-Risk customers, and potentially deprioritizing Hibernating customers in favor of higher-value segments.
# 
# ## 4. Category Preferences by Customer Segment

# In[7]:


def analyze_category_preferences_by_segment(df, customer_rfm):
    """
    Analyze product category preferences across different customer segments.

    Parameters:
        df (pd.DataFrame): The e-commerce dataset
        customer_rfm (pd.DataFrame): Customer metrics with RFM segments

    Returns:
        tuple: (segment_category_preferences DataFrame, bar chart figure)
    """
    # Extract customer segments
    customer_segments = customer_rfm[['Customer_Id', 'RFM_Segment']]

    # Merge with transaction data
    df_with_segments = pd.merge(df, customer_segments, on='Customer_Id', how='left')

    # Analyze category preferences by RFM segment
    segment_category_preferences = df_with_segments.groupby(['RFM_Segment', 'Product_Category'], observed=True).agg({
        'Total_Sales': 'sum',
        'Order_Id': 'nunique'
    }).reset_index()

    # Calculate percentage of segment sales by category
    segment_total_sales = df_with_segments.groupby('RFM_Segment', observed=True)['Total_Sales'].sum().reset_index()
    segment_category_preferences = pd.merge(
        segment_category_preferences,
        segment_total_sales,
        on='RFM_Segment',
        suffixes=('', '_Total')
    )

    segment_category_preferences['Category_Percentage'] = (segment_category_preferences['Total_Sales'] / 
                                                         segment_category_preferences['Total_Sales_Total'] * 100).round(1)

    # Create visualization for top segments
    top_segments = ['Champions', 'Loyal Customers', 'Big Spenders']
    top_segment_preferences = segment_category_preferences[segment_category_preferences['RFM_Segment'].isin(top_segments)]

    bar_fig = create_bar_chart(
        top_segment_preferences,
        x='RFM_Segment',
        y='Category_Percentage',
        color='Product_Category',
        title='Category Preferences of High-Value Customer Segments',
        x_label='Customer Segment',
        y_label='Percentage of Segment Sales (%)',
        barmode='stack'
    )

    # Create heatmap for all segments
    pivot_data = segment_category_preferences.pivot(index='RFM_Segment', 
                                                  columns='Product_Category', 
                                                  values='Category_Percentage')

    heatmap_fig = px.imshow(
        pivot_data,
        text_auto='.1f',
        labels=dict(x="Product Category", y="Customer Segment", color="Percentage (%)"),
        title="Category Preferences Across Customer Segments",
        color_continuous_scale=px.colors.sequential.Blues
    )

    heatmap_fig.update_layout(
        xaxis_title="Product Category",
        yaxis_title="Customer Segment"
    )

    return segment_category_preferences, bar_fig, heatmap_fig

# Run the analysis
segment_category_data, segment_category_bar, segment_category_heatmap = analyze_category_preferences_by_segment(df, rfm_customers)

# Display results
print("\nCategory Preferences by Customer Segment:")
display(format_display_table(segment_category_data.sort_values(['RFM_Segment', 'Category_Percentage'], ascending=[True, False])))

# Show charts
segment_category_bar.show()
segment_category_heatmap.show()


# **Key Insights on Category Preferences by Segment**
# Our analysis of category preferences across customer segments reveals distinct purchasing patterns:
# 
# 1. **Fashion Dominance Among High-Value Segments:** Champions (58.8%), Potential Loyalists (60.1%), and Big Spenders (63.9%) all show a strong preference for the Fashion category, closely aligned with our overall business focus.
# 2. **Segment-Specific Product Affinities:**
# 
# - At-Risk customers uniquely prefer Home & Furniture (56.9%) over Fashion (41.2%)
# - Hibernating customers show higher interest in Auto & Accessories (28.1%) compared to other segments
# - New Customers have above-average interest in Auto & Accessories (25.7%) as an entry category
# 
# 
# 3. **Electronics as a Secondary Category:** Electronics represents a consistent 3-10% of purchases across all segments, with Hibernating customers showing the highest relative interest (10.3%).
# 4. **Category Balance Among Loyal Customers:** The Loyal Customers segment shows the most balanced distribution across categories (Fashion 54.7%, Home & Furniture 24.8%, Auto & Accessories 16.1%), suggesting broader product exploration.
# 5. **Value-Based Category Selection:** Higher-value segments (Champions, Big Spenders) heavily favor Fashion, which aligns with our earlier finding that Fashion is our most profitable category.
# 
# These insights can directly inform our merchandising and marketing strategies, allowing for segment-targeted promotions and cross-selling opportunities. For example, At-Risk customers might respond best to Home & Furniture promotions, while attempts to reactivate Hibernating customers could focus on Auto & Accessories.
# 
# ## 5. Discount Sensitivity Analysis by Customer Segment

# In[8]:


def analyze_discount_sensitivity_by_segment(df, customer_rfm):
    """
    Analyze how different customer segments respond to discounts.

    Parameters:
        df (pd.DataFrame): The e-commerce dataset
        customer_rfm (pd.DataFrame): Customer metrics with RFM segments

    Returns:
        tuple: (segment_discount_sensitivity DataFrame, aov chart figure, margin chart figure)
    """
    # Extract customer segments
    customer_segments = customer_rfm[['Customer_Id', 'RFM_Segment']]

    # Merge with transaction data
    df_with_segments = pd.merge(df, customer_segments, on='Customer_Id', how='left')

    # Ensure discount bracket column exists
    if 'Discount_Bracket' not in df_with_segments.columns:
        df_with_segments['Discount_Bracket'] = pd.cut(
            df_with_segments['Discount'],
            bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0],
            labels=['0-10%', '11-20%', '21-30%', '31-40%', '41-50%', '51-100%'],
            include_lowest=True
        )

    # Analyze discount sensitivity by RFM segment
    segment_discount_sensitivity = df_with_segments.groupby(['RFM_Segment', 'Discount_Bracket'], observed=True).agg({
        'Total_Sales': 'sum',
        'Order_Id': 'nunique',
        'Total_Profit': 'sum'
    }).reset_index()

    # Calculate metrics
    segment_discount_sensitivity['Profit_Margin'] = (segment_discount_sensitivity['Total_Profit'] / 
                                                   segment_discount_sensitivity['Total_Sales'] * 100).round(1)
    segment_discount_sensitivity['Avg_Order_Value'] = (segment_discount_sensitivity['Total_Sales'] / 
                                                     segment_discount_sensitivity['Order_Id']).round(2)

    # Create visualizations for top segments
    top_segments = ['Champions', 'Loyal Customers', 'Big Spenders']
    top_segments_discount = segment_discount_sensitivity[
        segment_discount_sensitivity['RFM_Segment'].isin(top_segments)
    ]

    # Average order value by discount level
    aov_fig = create_line_chart(
        top_segments_discount,
        x='Discount_Bracket',
        y='Avg_Order_Value',
        color='RFM_Segment',
        title='Average Order Value by Discount Level for High-Value Segments',
        x_label='Discount Level',
        y_label='Average Order Value ($)',
        markers=True
    )

    # Profit margin by discount level
    margin_fig = create_line_chart(
        top_segments_discount,
        x='Discount_Bracket',
        y='Profit_Margin',
        color='RFM_Segment',
        title='Profit Margin by Discount Level for High-Value Segments',
        x_label='Discount Level',
        y_label='Profit Margin (%)',
        markers=True
    )

    return segment_discount_sensitivity, aov_fig, margin_fig

# Run the analysis
discount_sensitivity_data, aov_chart, margin_chart = analyze_discount_sensitivity_by_segment(df, rfm_customers)

# Display results
print("\nDiscount Sensitivity by Customer Segment:")
display(format_display_table(discount_sensitivity_data))

# Show charts
aov_chart.show()
margin_chart.show()


# **Key Insights on Discount Sensitivity by Segment**
# Our discount sensitivity analysis reveals distinct patterns in how different customer segments respond to promotional offers:
# 
# 1. **Champions Show Strong Discount Response:** Champions exhibit a clear positive correlation between discount level and average order value, rising from \$442.42 at 0-10% discount to \$486.53 at 41-50% discount, suggesting higher discount tiers effectively drive increased spending in this segment.
# 2. **Big Spenders Respond to Mid-Level Discounts:** Big Spenders show the highest average order value ($676.85) at the 21-30% discount tier, indicating a sweet spot for this segment that balances discount depth with spending motivation.
# 3. **Consistent Margin Impact Across Segments:** All segments show a similar pattern of margin reduction with increased discount levels, with the steepest drop occurring between the 0-10% and 21-30% tiers.
# 4. **Higher Baseline Margins for High-Value Segments:** Champions and Potential Loyalists maintain higher profit margins (51.3% and 52.1% respectively) at the lowest discount tier (0-10%) compared to other segments.
# 5. **Discount Recovery at Higher Tiers:** Interestingly, all segments show a slight profit margin recovery at the 31-40% and 41-50% discount tiers compared to the 21-30% tier, suggesting customers may purchase higher-margin products at these deeper discount levels.
# 
# These insights can directly inform segment-specific promotional strategies, such as offering Champions higher discount tiers to maximize order value, targeting Big Spenders with 21-30% promotions for optimal balance, and potentially limiting discounts for segments with lower margin sensitivity.
# 
# ## 6. Device and Payment Method Preferences by Segment

# In[9]:


def analyze_device_preferences_by_segment(df, customer_rfm):
    """
    Analyze device preferences across different customer segments.

    Parameters:
        df (pd.DataFrame): The e-commerce dataset
        customer_rfm (pd.DataFrame): Customer metrics with RFM segments

    Returns:
        tuple: (device_segment_preferences DataFrame, chart figure)
    """
    # Extract customer segments
    customer_segments = customer_rfm[['Customer_Id', 'RFM_Segment']]

    # Merge with transaction data
    df_with_segments = pd.merge(df, customer_segments, on='Customer_Id', how='left')

    # Analyze device preference by segment
    device_segment_preferences = df_with_segments.groupby(['RFM_Segment', 'Device_Type'], observed=True).agg({
        'Total_Sales': 'sum',
        'Customer_Id': 'nunique'
    }).reset_index()

    # Calculate percentage of customers within segment using each device
    segment_total_customers = df_with_segments.groupby('RFM_Segment', observed=True)['Customer_Id'].nunique().reset_index()
    device_segment_preferences = pd.merge(
        device_segment_preferences,
        segment_total_customers,
        on='RFM_Segment',
        suffixes=('', '_Total')
    )

    device_segment_preferences['Customer_Percentage'] = (device_segment_preferences['Customer_Id'] / 
                                                       device_segment_preferences['Customer_Id_Total'] * 100).round(1)

    # Create visualization
    fig = px.bar(
        device_segment_preferences[device_segment_preferences['Device_Type'] == 'Mobile'],
        x='RFM_Segment',
        y='Customer_Percentage',
        title='Mobile Usage by Customer Segment',
        labels={'RFM_Segment': 'Customer Segment', 'Customer_Percentage': 'Percentage of Mobile Users (%)'},
        color='Customer_Percentage',
        color_continuous_scale=px.colors.sequential.Viridis,
        text=device_segment_preferences[device_segment_preferences['Device_Type'] == 'Mobile']['Customer_Percentage'].apply(lambda x: f"{x:.1f}%")
    )

    fig.update_traces(textposition='outside')
    fig.update_layout(xaxis_tickangle=-45)

    return device_segment_preferences, fig

def analyze_payment_preferences_by_segment(df, customer_rfm):
    """
    Analyze payment method preferences across different customer segments.

    Parameters:
        df (pd.DataFrame): The e-commerce dataset
        customer_rfm (pd.DataFrame): Customer metrics with RFM segments

    Returns:
        tuple: (payment_segment_preferences DataFrame, chart figure)
    """
    # Extract customer segments
    customer_segments = customer_rfm[['Customer_Id', 'RFM_Segment']]

    # Merge with transaction data
    df_with_segments = pd.merge(df, customer_segments, on='Customer_Id', how='left')

    # Analyze payment method preferences by segment
    payment_segment_preferences = df_with_segments.groupby(['RFM_Segment', 'Payment_method'], observed=True).agg({
        'Total_Sales': 'sum',
        'Order_Id': 'nunique'
    }).reset_index()

    # Calculate percentage of segment sales by payment method
    segment_total_sales = df_with_segments.groupby('RFM_Segment', observed=True)['Total_Sales'].sum().reset_index()
    payment_segment_preferences = pd.merge(
        payment_segment_preferences,
        segment_total_sales,
        on='RFM_Segment',
        suffixes=('', '_Total')
    )

    payment_segment_preferences['Sales_Percentage'] = (payment_segment_preferences['Total_Sales'] / 
                                                     payment_segment_preferences['Total_Sales_Total'] * 100).round(1)

    # Create visualization for top segments
    top_segments = ['Champions', 'Loyal Customers', 'Big Spenders']
    top_segments_payment = payment_segment_preferences[
        payment_segment_preferences['RFM_Segment'].isin(top_segments)
    ]

    fig = create_bar_chart(
        top_segments_payment,
        x='RFM_Segment',
        y='Sales_Percentage',
        color='Payment_method',
        title='Payment Method Preferences of High-Value Customer Segments',
        x_label='Customer Segment',
        y_label='Percentage of Segment Sales (%)',
        barmode='stack'
    )

    return payment_segment_preferences, fig

# Run the device analysis
device_preferences_data, device_chart = analyze_device_preferences_by_segment(df, rfm_customers)

# Run the payment analysis
payment_preferences_data, payment_chart = analyze_payment_preferences_by_segment(df, rfm_customers)

# Display results
print("\nDevice Preferences by Customer Segment:")
display(format_display_table(device_preferences_data))

print("\nPayment Method Preferences by Customer Segment:")
display(format_display_table(payment_preferences_data))

# Show charts
device_chart.show()
payment_chart.show()


# **Key Insights on Device and Payment Preferences**
# Our analysis of device and payment preferences across customer segments reveals meaningful patterns in how different segments engage with our platform:
# 
# 1. **Mobile Usage Highest Among Champions:** Champions show the highest mobile adoption at 14.4%, significantly above the average, suggesting these high-value customers embrace multi-channel shopping.
# 2. **Potential Loyalists Leverage Mobile:** Potential Loyalists show the second-highest mobile usage at 11.3%, indicating potential for mobile-focused retention strategies for this valuable segment.
# 3. **Limited Mobile Adoption Among Value Segments:** Big Spenders (4.7%), Hibernating (5.9%), and New Customers (6.1%) show the lowest mobile usage, suggesting the mobile experience may not be effectively engaging these segments.
# 4. **Consistent Payment Method Distribution:** All segments show remarkably similar payment method preferences, with Credit Card dominating at approximately 73-76% across all segments, followed by Money Order (17-20%), and minimal usage of E-Wallet (4-6%) and Debit Card (1-2%).
# 5. **Minor Payment Preference Variations:** Potential Loyalists show slightly higher Money Order usage (19.7%) compared to other segments, while Big Spenders have marginally lower E-Wallet adoption (4.9%).
# 
# These insights suggest that while payment preferences are relatively consistent across segments, device preferences show significant variation. The higher mobile adoption among Champions indicates an opportunity to enhance mobile experiences for other segments, particularly Big Spenders who could potentially increase purchase frequency through improved mobile engagement.
# 
# ## 7. Combined Profitability Analysis by Customer Segment

# In[10]:


def analyze_segment_profitability(df, customer_rfm):
    """
    Analyze profitability metrics across different customer segments.

    Parameters:
        df (pd.DataFrame): The e-commerce dataset
        customer_rfm (pd.DataFrame): Customer metrics with RFM segments

    Returns:
        tuple: (segment_profitability DataFrame, chart figure)
    """
    # Extract customer segments
    customer_segments = customer_rfm[['Customer_Id', 'RFM_Segment']]

    # Merge with transaction data
    df_with_segments = pd.merge(df, customer_segments, on='Customer_Id', how='left')

    # Calculate metrics by RFM segment
    segment_profitability = df_with_segments.groupby('RFM_Segment', observed=True).agg({
        'Total_Sales': 'sum',
        'Total_Profit': 'sum',
        'Shipping_Cost': 'sum',
        'Discount_Amount': 'sum',
        'Customer_Id': 'nunique'
    }).reset_index()

    # Calculate derived metrics
    segment_profitability['Profit_Margin'] = (segment_profitability['Total_Profit'] / 
                                            segment_profitability['Total_Sales'] * 100).round(1)
    segment_profitability['Revenue_per_Customer'] = (segment_profitability['Total_Sales'] / 
                                                   segment_profitability['Customer_Id']).round(2)
    segment_profitability['Profit_per_Customer'] = (segment_profitability['Total_Profit'] / 
                                                  segment_profitability['Customer_Id']).round(2)
    segment_profitability['Shipping_Percentage'] = (segment_profitability['Shipping_Cost'] / 
                                                  segment_profitability['Total_Sales'] * 100).round(2)
    segment_profitability['Discount_Percentage'] = (segment_profitability['Discount_Amount'] / 
                                                  segment_profitability['Total_Sales'] * 100).round(2)

    # Sort by profit per customer
    segment_profitability = segment_profitability.sort_values('Profit_per_Customer', ascending=False)

    # Create visualization
    fig = create_bar_chart(
        segment_profitability,
        x='RFM_Segment',
        y='Profit_per_Customer',
        color='Profit_Margin',
        title='Profit per Customer by Segment',
        x_label='Customer Segment',
        y_label='Profit per Customer ($)',
        color_continuous_scale=px.colors.sequential.Viridis,
        text=segment_profitability['Profit_per_Customer'].apply(lambda x: f'${x:.2f}')
    )

    # Create bubble chart
    bubble_fig = px.scatter(
        segment_profitability,
        x='Revenue_per_Customer',
        y='Profit_Margin',
        size='Total_Sales',
        color='RFM_Segment',
        hover_name='RFM_Segment',
        title='Segment Profitability Analysis',
        labels={
            'Revenue_per_Customer': 'Revenue per Customer ($)',
            'Profit_Margin': 'Profit Margin (%)',
            'Total_Sales': 'Total Revenue'
        },
        size_max=50
    )

    return segment_profitability, fig, bubble_fig

# Run the analysis
segment_profitability_data, profitability_chart, bubble_chart = analyze_segment_profitability(df, rfm_customers)

# Display results
print("\nProfitability Analysis by Customer Segment:")
display(format_display_table(segment_profitability_data))

# Show charts
profitability_chart.show()
bubble_chart.show()


# **Key Insights on Segment Profitability**
# Our combined profitability analysis reveals critical differences in the financial value of different customer segments:
# 
# 1. **Champions Deliver Highest Customer Value:** Champions generate \$422.58 profit per customer, more than double the value of Loyal Customers (\$193.04) and significantly higher than all other segments, confirming their status as our most valuable customer group.
# 2. **Potential Loyalists Show Exceptional Efficiency:** Potential Loyalists maintain the highest profit margin (45.3%) among all segments and deliver the second-highest profit per customer (\$337.73), suggesting excellent optimization of this segment.
# 3. **Big Spenders Show Margin Potential:** Big Spenders generate strong profit per customer (\$288.50) despite receiving the highest average discount (32.92%), indicating potential for margin improvement through optimized discount strategies.
# 4. **Lower-Tier Segment Value Gap:** A significant value gap exists between top segments and others, with "Need Attention," "New Customers," and "Hibernating" segments generating less than $85 profit per customer.
# 5. **Shipping Cost Impact by Segment:** Lower-value segments bear a disproportionate shipping cost burden, with New Customers (3.45%), Hibernating (3.42%), and Need Attention (3.33%) facing shipping costs as a percentage of sales more than double that of Champions (1.64%) and Potential Loyalists (1.56%).
# 
# These insights highlight the importance of segment-specific strategies: retention and expansion for Champions and Potential Loyalists, discount optimization for Big Spenders, and potentially reconsidering investment in lower-value segments given their significantly lower profitability and higher relative costs.
# 
# ## 8. Executive Dashboard and Recommendations

# In[11]:


def create_customer_segment_dashboard(df, customer_rfm, segment_profitability):
    """
    Create an interactive dashboard showcasing key customer segment insights.

    Parameters:
        df (pd.DataFrame): The e-commerce dataset
        customer_rfm (pd.DataFrame): Customer metrics with RFM segments
        segment_profitability (pd.DataFrame): Profitability metrics by segment

    Returns:
        plotly.graph_objects.Figure: Dashboard figure
    """
    # Extract top segments for detailed analysis
    top_segments = ['Champions', 'Loyal Customers', 'Big Spenders']

    # Create dashboard layout
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Revenue Contribution by RFM Segment',
            'Profit per Customer by Segment',
            'Customer Segment Distribution',
            'Customer Value Matrix'
        ),
        specs=[
            [{"type": "pie"}, {"type": "bar"}],
            [{"type": "pie"}, {"type": "scatter"}]
        ]
    )

    # 1. Revenue contribution (pie chart)
    fig.add_trace(
        go.Pie(
            labels=segment_profitability['RFM_Segment'],
            values=segment_profitability['Total_Sales'],
            textinfo='percent+label',
            hole=0.4
        ),
        row=1, col=1
    )

    # 2. Profit per customer (bar chart)
    fig.add_trace(
        go.Bar(
            x=segment_profitability['RFM_Segment'],
            y=segment_profitability['Profit_per_Customer'],
            text=[f"${p:.2f}" for p in segment_profitability['Profit_per_Customer']],
            textposition='outside'
        ),
        row=1, col=2
    )

    # 3. Customer distribution (pie chart)
    fig.add_trace(
        go.Pie(
            labels=segment_profitability['RFM_Segment'],
            values=segment_profitability['Customer_Id'],
            textinfo='percent+label',
            hole=0.4
        ),
        row=2, col=1
    )

    # 4. Customer value matrix (scatter plot)
    # Use frequency and monetary value from RFM
    customer_counts = customer_rfm.groupby(['F_Score', 'M_Score']).size().reset_index(name='Count')

    fig.add_trace(
        go.Scatter(
            x=customer_counts['F_Score'],
            y=customer_counts['M_Score'],
            mode='markers',
            marker=dict(
                size=customer_counts['Count'],
                sizemode='area',
                sizeref=2.*max(customer_counts['Count'])/(40.**2),
                sizemin=4,
                color=customer_counts['Count'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Customer Count")
            ),
            text=[f"F: {f}, M: {m}, Count: {c}" for f, m, c in zip(
                customer_counts['F_Score'], 
                customer_counts['M_Score'], 
                customer_counts['Count']
            )],
            hoverinfo='text'
        ),
        row=2, col=2
    )

    # Update layout and axes
    fig.update_xaxes(title_text="Frequency Score", row=2, col=2)
    fig.update_yaxes(title_text="Monetary Score", row=2, col=2)
    fig.update_xaxes(title_text="Customer Segment", row=1, col=2)
    fig.update_yaxes(title_text="Profit per Customer ($)", row=1, col=2)

    fig.update_layout(
        title_text="Customer Segmentation Dashboard",
        height=800,
        width=1200,
        showlegend=False
    )

    return fig

def display_strategic_recommendations():
    """Display strategic recommendations based on customer segmentation analysis."""

    recommendations_html = """
    <div style="background-color:#f8f9fa; padding:20px; border-radius:10px; margin-top:30px;">
        <h2 style="color:#333; border-bottom:2px solid #ddd; padding-bottom:10px;">Strategic Recommendations: Customer-Focused Strategies</h2>

        <div style="display:flex; flex-wrap:wrap; justify-content:space-between;">
            <div style="width:48%; background-color:#e8f4f8; padding:15px; border-radius:8px; margin-bottom:15px;">
                <h3 style="color:#0066cc;">1. Champions Retention Program</h3>
                <p>Develop a VIP program for Champions (22.9% of customers, 43.1% of revenue) to maximize retention and lifetime value.</p>
                <p><strong>Action Steps:</strong> Implement exclusive benefits, early access to new products, personalized recommendations focused on Fashion (58.8% of their purchases), and mobile app enhancements (14.4% already use mobile).</p>
            </div>

            <div style="width:48%; background-color:#e8f8f4; padding:15px; border-radius:8px; margin-bottom:15px;">
                <h3 style="color:#00994d;">2. Frequency Acceleration for Big Spenders</h3>
                <p>Create targeted strategies to increase purchase frequency for Big Spenders who have high order values ($655.19) but low purchase frequency (1.0).</p>
                <p><strong>Action Steps:</strong> Implement replenishment reminders, limited-time offers optimized at their most effective 21-30% discount tier, fashion-focused promotions, and improved mobile experience (only 4.7% currently use mobile).</p>
            </div>

            <div style="width:48%; background-color:#f8f4e8; padding:15px; border-radius:8px; margin-bottom:15px;">
                <h3 style="color:#cc7a00;">3. One-Time Customer Conversion Initiative</h3>
                <p>Develop strategies to convert more one-time purchasers (73.9% of customers) to repeat buyers, as each frequency increase significantly boosts customer value.</p>
                <p><strong>Action Steps:</strong> Implement post-purchase follow-up program, first-to-second purchase incentives, and personalized product recommendations based on initial purchase category.</p>
            </div>

            <div style="width:48%; background-color:#f4e8f8; padding:15px; border-radius:8px; margin-bottom:15px;">
                <h3 style="color:#7a00cc;">4. At-Risk Customer Reactivation</h3>
                <p>Create targeted reactivation campaigns for At-Risk customers focused on their unique category preferences (Home & Furniture at 56.9% of purchases).</p>
                <p><strong>Action Steps:</strong> Develop Home & Furniture specific promotions, optimize shipping costs (currently 2.37% of sales for this segment), and implement a win-back email campaign with targeted offers.</p>
            </div>
        </div>

        <h3 style="color:#333; margin-top:20px;">Implementation Priority:</h3>
        <ol>
            <li><strong>High Impact/Effort Ratio:</strong> Champions Retention (43.1% of revenue)</li>
            <li><strong>Medium Impact/Effort Ratio:</strong> Big Spenders Frequency Acceleration (14.3% of revenue)</li>
            <li><strong>Medium Impact/High Effort:</strong> One-Time Customer Conversion (55.9% of revenue but high resource requirement)</li>
            <li><strong>Lower Impact/Effort Ratio:</strong> At-Risk Reactivation (5.3% of revenue)</li>
        </ol>

        <p style="font-style:italic; margin-top:20px;">This segment-based approach focuses resources on the highest-value customer groups while implementing targeted strategies for growth segments, maximizing return on marketing investment while improving overall customer experience.</p>
    </div>
    """

    return HTML(recommendations_html)

# Create the executive dashboard
segment_dashboard = create_customer_segment_dashboard(df, rfm_customers, segment_profitability_data)

# Show the dashboard
segment_dashboard.show()

# Display strategic recommendations
display_strategic_recommendations()


# In[ ]:





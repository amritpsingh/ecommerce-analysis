#!/usr/bin/env python
# coding: utf-8

# # E-commerce Sales Performance Analysis
# 
# ### Executive Summary
# This analysis examines our e-commerce platform's sales performance across products, categories, and time periods to identify growth drivers and patterns. Our findings reveal:
# 
# Fashion dominates revenue (57.5%) with T-Shirts as the top-selling product
# Strong seasonality with Fall generating 30.8% of annual revenue
# High sales concentration with the top 10 products contributing 58.7% of total revenue
# Weekday purchasing patterns showing higher activity Tuesday-Thursday
# Significant monthly fluctuations with November peak sales and February trough
# 
# The insights and recommendations in this notebook will help optimize inventory planning, marketing campaigns, and promotional strategies throughout the year.
# 
# ### Introduction
# Understanding sales performance patterns is crucial for optimizing business operations in e-commerce. This analysis aims to:
# 
# Identify top-performing products and categories driving revenue
# Uncover temporal patterns (monthly, weekly, daily, hourly)
# Analyze seasonal trends and their impact on different product categories
# Provide actionable recommendations for business growth
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
from ecommerce_utils import (load_data, format_display_table, bar_chart, line_chart)


# In[2]:


# Load the data
df = load_data('ecommerce_cleaned.parquet')
df.head()


# ## 1. Category and Product Performance Analysis
# ### 1.1 Revenue by Product Category

# In[3]:


def analyze_category_performance(df):
    """
    Analyze sales performance by product category.

    Parameters:
        df (pd.DataFrame): The e-commerce dataset

    Returns:
        tuple: (category_revenue DataFrame, plotly figure)
    """
    # Group data by product category and calculate total revenue
    category_revenue = df.groupby('Product_Category')['Total_Sales'].sum().sort_values(ascending=False).reset_index()

    # Calculate percentage contribution of each category
    category_revenue['Percentage'] = (category_revenue['Total_Sales'] / category_revenue['Total_Sales'].sum() * 100).round(1)

    # Create visualization
    fig = bar_chart(
        x='Product_Category',
        y='Total_Sales',
        df=category_revenue,
        plot_title='Revenue by Product Category',
        x_label='Product Category',
        y_label='Total Revenue ($)',
        percentage_col='Percentage',
        scale='millions'
    )

    return category_revenue, fig

# Run the analysis
category_data, category_chart = analyze_category_performance(df)
category_chart.show()

# Display the data table
format_display_table(category_data)


# **Key Insight: Fashion Dominates Revenue**
# - Fashion is the clear revenue leader, generating \$11.3M (57.5% of total revenue), more than twice as much as the next category.
# - Home & Furniture is a strong secondary category at \$4.9M (24.9%)
# - Electronics generates surprisingly little revenue (\$0.9M, 4.4%) despite typically being a high-ticket category in e-commerce.
# 
# This distribution suggests our platform is primarily fashion-focused with home goods as a complementary offering. The relatively low electronics revenue might indicate either limited product offerings, lack of competitive pricing, or an opportunity for expansion.
# 
# ### 1.2 Top Revenue-Generating Products

# In[4]:


def analyze_top_products(df, top_n=10):
    """
    Identify and analyze top-selling products.

    Parameters:
        df (pd.DataFrame): The e-commerce dataset
        top_n (int): Number of top products to analyze

    Returns:
        tuple: (top_products DataFrame, plotly figure)
    """
    # Identify top products by revenue
    top_products = df.groupby('Product')['Total_Sales'].sum().sort_values(ascending=False).head(top_n).reset_index()

    # Calculate percentage contribution of each product
    top_products['Percentage'] = (top_products['Total_Sales'] / df['Total_Sales'].sum() * 100).round(1)

    # Calculate cumulative percentage
    top_products['Cumulative_Percentage'] = top_products['Percentage'].cumsum().round(1)

    # Create visualization
    fig = bar_chart(
        x='Product',
        y='Total_Sales',
        df=top_products,
        plot_title=f'Top {top_n} Products by Revenue',
        x_label='Product',
        y_label='Total Revenue ($)',
        percentage_col='Percentage',
        scale='millions'
    )

    # Add a text annotation about revenue concentration
    total_percentage = top_products['Percentage'].sum().round(1)
    annotation = dict(
        x=0.5,
        y=-0.15,
        xref='paper',
        yref='paper',
        text=f'These {top_n} products represent {total_percentage}% of total revenue',
        showarrow=False,
        font=dict(size=12),
        align='center'
    )
    fig.update_layout(annotations=[annotation])

    return top_products, fig

# Run the analysis
top_products_data, top_products_chart = analyze_top_products(df)
top_products_chart.show()

# Display the data table
format_display_table(top_products_data)


# **Key Insight: High Revenue Concentration in Top Products**
# Our top 10 products alone contribute 58.7% of total revenue, indicating a high dependence on a small product selection. 
# 
# - T-Shirts lead with \$1.5M (7.6%), followed closely by Titak Watch at \$1.4M (7.3%) and Running Shoes at \$1.4M (6.9%).
# - The dominance of apparel and accessories (T-Shirts, Watches, Shoes, Jeans) in our top sellers aligns with our fashion-focused business model.
# - Watches are particularly strong performers, with Titak and Fossil watches combined accounting for 12.2% of revenue, suggesting we've developed a strong reputation in this category.
# 
# This concentration highlights both a strength (clear product focus) and risk (dependency on few revenue drivers). Diversification strategies should balance building on these strengths while reducing revenue concentration risk.
# 
# ## 2. Temporal Sales Analysis
# ### 2.1 Monthly Sales Trends

# In[5]:


def analyze_monthly_sales(df):
    """
    Analyze monthly sales patterns and growth rates.

    Parameters:
        df (pd.DataFrame): The e-commerce dataset

    Returns:
        tuple: (monthly_sales DataFrame, trend figure, growth figure)
    """
    # Group by month and sum total sales
    monthly_sales = (df.groupby(df['Order_Date'].dt.to_period('M'))['Total_Sales']
                      .sum()
                      .reset_index()
                      .rename(columns={'Order_Date': 'Month_Period'}))

    # Add date components and formatting
    monthly_sales['Order_Date'] = monthly_sales['Month_Period'].dt.to_timestamp()
    monthly_sales['Month'] = monthly_sales['Month_Period'].dt.strftime('%b')
    monthly_sales['Month_Num'] = monthly_sales['Month_Period'].dt.month

    # Ensure proper ordering by month number
    monthly_sales = monthly_sales.sort_values('Month_Num').reset_index(drop=True)

    # Calculate month-over-month growth
    monthly_sales['MoM_Growth'] = monthly_sales['Total_Sales'].pct_change() * 100

    # Create label columns for annotations
    monthly_sales['Sales_Label'] = monthly_sales['Total_Sales'].apply(lambda v: f'${v:,.0f}')
    monthly_sales['Growth_Label'] = monthly_sales['MoM_Growth'].apply(
        lambda v: f'{v:.1f}%' if pd.notnull(v) else '')

    # Create monthly sales trend visualization
    trend_fig = line_chart(
        x='Month',
        y='Total_Sales',
        df=monthly_sales,
        plot_title='Monthly Sales Trend',
        x_label='Month',
        y_label='Total Revenue ($)',
        text_col='Sales_Label',
        highlight_max=True,
        highlight_min=True
    )

    # Force x-axis to follow the sorted month order
    trend_fig.update_xaxes(categoryorder='array', categoryarray=monthly_sales['Month'].tolist())

    # Create MoM growth visualization
    growth_df = monthly_sales.iloc[1:].copy()  # Exclude first month (no growth rate)

    growth_fig = px.bar(
        growth_df,
        x='Month',
        y='MoM_Growth',
        color='MoM_Growth',
        title='Month-over-Month Sales Growth (%)',
        labels={'Month': 'Month', 'MoM_Growth': 'Growth Rate (%)'},
        template='plotly_white',
        color_continuous_scale=px.colors.diverging.BrBG,
        text='Growth_Label'
    )

    # Add a horizontal line at y=0
    growth_fig.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3)

    # Update layout
    growth_fig.update_traces(textposition='outside')
    growth_fig.update_layout(
        xaxis_title='Month',
        yaxis_title='Growth Rate (%)',
        xaxis_tickangle=-45
    )

    # Calculate summary statistics
    highest = monthly_sales.loc[monthly_sales['Total_Sales'].idxmax()]
    lowest = monthly_sales.loc[monthly_sales['Total_Sales'].idxmin()]
    avg_sales = monthly_sales['Total_Sales'].mean()

    print("Monthly Sales Summary:")
    print(f"Month with highest sales: {highest['Month']} (${highest['Total_Sales']:,.2f})")
    print(f"Month with lowest sales: {lowest['Month']} (${lowest['Total_Sales']:,.2f})")
    print(f"Average monthly sales: ${avg_sales:,.2f}")

    return monthly_sales, trend_fig, growth_fig

# Run the analysis
monthly_data, monthly_trend_chart, monthly_growth_chart = analyze_monthly_sales(df)

# Display the visualizations
monthly_trend_chart.show()
monthly_growth_chart.show()

# Display the data table
format_display_table(monthly_data)


# **Key Insight: Distinct Seasonal Cycles with Volatile Growth**
# Our e-commerce platform exhibits clear seasonal patterns with significant monthly volatility. November represents our peak sales month at \$2.23M, while February shows our lowest performance at \$842K, less than half of the peak. The average monthly sales of \$1.64M suggests we have solid baseline revenue, but with significant fluctuations.
# The growth rate chart reveals a business with pronounced cycles:
# 
# - Strong growth period from February to May (peaking at +40.6% in April)
# - Mid-year correction in June (-21.7%)
# - Summer recovery from June to July (+25.4%)
# - Fall growth in September through November
# - December decline (-13.9%) that's surprising given typical holiday shopping patterns
# 
# These cyclical patterns should inform inventory planning and marketing campaigns, with particular attention to supporting the critical February-May growth period and investigating the December decline that contradicts industry norms.
# 
# ### 2.2 Weekly Sales Patterns

# In[6]:


def analyze_weekly_sales(df):
    """
    Analyze weekly sales patterns throughout the year.

    Parameters:
        df (pd.DataFrame): The e-commerce dataset

    Returns:
        tuple: (weekly_sales DataFrame, plotly figure)
    """
    # Create a copy to avoid modifying the original
    df_weekly = df.copy()

    # Ensure we have Year and Week columns
    df_weekly['Year'] = df_weekly['Order_DateTime'].dt.year
    df_weekly['Week'] = df_weekly['Order_DateTime'].dt.isocalendar().week

    # Group by year and week
    weekly_sales = df_weekly.groupby(['Year', 'Week'])['Total_Sales'].sum().reset_index()

    # Create YearWeek column for better display
    weekly_sales['YearWeek'] = (weekly_sales['Year'].astype(str) + '-W' + 
                               weekly_sales['Week'].astype(str).str.zfill(2))

    # Sort by year and week
    weekly_sales = weekly_sales.sort_values(['Year', 'Week']).reset_index(drop=True)

    # Find highest and lowest weeks
    high_week = weekly_sales.loc[weekly_sales['Total_Sales'].idxmax()]
    low_week = weekly_sales.loc[weekly_sales['Total_Sales'].idxmin()]

    # Create visualization
    fig = px.line(
        weekly_sales,
        x='YearWeek',
        y='Total_Sales',
        title='Weekly Sales Trend',
        labels={'YearWeek': 'Year-Week', 'Total_Sales': 'Total Revenue ($)'},
        template='plotly_white'
    )

    # Update line appearance
    fig.update_traces(line=dict(width=2), marker=dict(size=6))

    # Highlight peak and trough weeks
    fig.add_scatter(
        x=[high_week['YearWeek']],
        y=[high_week['Total_Sales']],
        mode='markers+text',
        marker=dict(color='green', size=12),
        text=['Peak'],
        textposition='top center',
        name=f"Peak: ${high_week['Total_Sales']:,.0f}"
    )

    fig.add_scatter(
        x=[low_week['YearWeek']],
        y=[low_week['Total_Sales']],
        mode='markers+text',
        marker=dict(color='red', size=12),
        text=['Trough'],
        textposition='bottom center',
        name=f"Trough: ${low_week['Total_Sales']:,.0f}"
    )

    # Set x-ticks to show every 4th week for clarity
    tickvals = weekly_sales['YearWeek'].iloc[::4].tolist()
    fig.update_xaxes(tickmode='array', tickvals=tickvals, tickangle=45)

    # Add annotations about key weeks
    fig.update_layout(
        annotations=[
            dict(
                x=high_week['YearWeek'],
                y=high_week['Total_Sales'],
                xref="x",
                yref="y",
                text=f"Week {high_week['Week']}: ${high_week['Total_Sales']:,.0f}",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40
            ),
            dict(
                x=low_week['YearWeek'],
                y=low_week['Total_Sales'],
                xref="x",
                yref="y",
                text=f"Week {low_week['Week']}: ${low_week['Total_Sales']:,.0f}",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=40
            )
        ]
    )

    return weekly_sales, fig

# Run the analysis
weekly_data, weekly_chart = analyze_weekly_sales(df)
weekly_chart.show()

# Show top 5 and bottom 5 weeks
top_weeks = weekly_data.nlargest(5, 'Total_Sales')
bottom_weeks = weekly_data.nsmallest(5, 'Total_Sales')

print("Top 5 weeks by sales:")
display(format_display_table(top_weeks))

print("Bottom 5 weeks by sales:")
display(format_display_table(bottom_weeks))


# **Key Insight: Weekly Analysis Reveals Hidden Patterns**
# The weekly sales analysis provides more granular insights that validate our monthly trends while revealing specific periods worthy of further investigation:
# 
# - **Extraordinary Peak in Week 17:** The $726K spike in this week (late April) is dramatically higher than surrounding periods and demands investigation. This could be due to a successful promotion, product launch, or external event.
# - **Strong November Performance:** Weeks 44-46 (early-mid November) show consistently strong performance, reinforcing our monthly finding about November's importance.
# - **February Weakness Confirmed:** The lowest sales appear in weeks 5-6 and 12, validating our monthly finding about February's weakness.
# 
# This weekly analysis helps pinpoint exactly when our peaks and troughs occur, allowing for more targeted planning of promotions, inventory, and marketing campaigns. The week 17 anomaly merits special attention as understanding its cause could help replicate its success.
# 
# ### 2.3 Day-of-Week Analysis

# In[7]:


def analyze_day_of_week_sales(df):
    """
    Analyze sales patterns by day of week.

    Parameters:
        df (pd.DataFrame): The e-commerce dataset

    Returns:
        tuple: (day_sales DataFrame, plotly figure)
    """
    # Create a copy to avoid modifying the original
    df_daily = df.copy()

    # Ensure Weekday column exists
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    if 'Weekday' not in df_daily.columns:
        df_daily['Weekday'] = df_daily['Order_DateTime'].dt.day_name()

    # Group sales by weekday and reindex to enforce day order
    day_sales = df_daily.groupby('Weekday')['Total_Sales'].sum().reindex(day_order).reset_index()

    # Calculate percentage of total
    day_sales['Percentage'] = (day_sales['Total_Sales'] / day_sales['Total_Sales'].sum() * 100).round(1)

    # Create visualization
    fig = bar_chart(
        x='Weekday',
        y='Total_Sales',
        df=day_sales,
        plot_title='Sales by Day of Week',
        x_label='Day of Week',
        y_label='Total Revenue ($)',
        percentage_col='Percentage',
        scale='millions',
        xtickangle=0,
        color='Total_Sales',
        color_continuous_scale=px.colors.diverging.RdBu,
        color_continuous_midpoint=0
    )

    return day_sales, fig

# Run the analysis
day_data, day_chart = analyze_day_of_week_sales(df)
day_chart.show()

# Display the data table
format_display_table(day_data)


# **Key Insight: Weekday Shopping Dominates**
# Our analysis reveals a clear weekday shopping preference among our customers, with Tuesday generating the highest revenue at $3.04M (15.5% of total). Sales gradually decrease through the week, with weekend days showing lower performance than weekdays.
# 
# This pattern contradicts common e-commerce trends where weekends often show higher activity. It suggests our platform attracts customers who shop during work hours, potentially indicating:
# 
# - **Professional Customer Base:** Our shoppers may be professionals browsing during work breaks
# - **Office-Based Shopping:** Customers might prefer making purchases from their work computers
# - **Weekday Routine:** Our audience may include stay-at-home parents or others who shop while children are at school
# 
# This insight is valuable for timing promotional emails, social media campaigns, and new product launches. Marketing activities scheduled for Monday and Tuesday mornings could be particularly effective at driving sales.
# 
# ### 2.4 Hourly Sales Distribution

# In[8]:


def analyze_hourly_sales(df):
    """
    Analyze sales patterns by hour of day.

    Parameters:
        df (pd.DataFrame): The e-commerce dataset

    Returns:
        tuple: (hour_sales DataFrame, plotly figure)
    """
    # Create a copy to avoid modifying the original
    df_hourly = df.copy()

    # Ensure Hour column exists
    if 'Hour' not in df_hourly.columns:
        df_hourly['Hour'] = df_hourly['Order_DateTime'].dt.hour

    # Group by hour
    hour_sales = df_hourly.groupby('Hour')['Total_Sales'].sum().reset_index()

    # Calculate percentage of total
    hour_sales['Percentage'] = (hour_sales['Total_Sales'] / hour_sales['Total_Sales'].sum() * 100).round(1)

    # Calculate average vs. off-hour
    avg_sales = hour_sales['Total_Sales'].mean()
    hour_sales['vs_Average'] = ((hour_sales['Total_Sales'] / avg_sales) - 1) * 100

    # Categorize hours
    def categorize_hour(hour):
        if 0 <= hour < 6:
            return 'Night (12AM-6AM)'
        elif 6 <= hour < 12:
            return 'Morning (6AM-12PM)'
        elif 12 <= hour < 18:
            return 'Afternoon (12PM-6PM)'
        else:
            return 'Evening (6PM-12AM)'

    hour_sales['Day_Period'] = hour_sales['Hour'].apply(categorize_hour)

    # Create visualization
    fig = bar_chart(
        x='Hour',
        y='Total_Sales',
        df=hour_sales,
        plot_title='Sales by Hour of Day',
        x_label='Hour (24-hour format)',
        y_label='Total Revenue ($)',
        percentage_col='Percentage',
        scale='millions',
        xtickangle=0,
        color='vs_Average',
        color_continuous_scale=px.colors.diverging.RdBu,
        color_continuous_midpoint=0
    )

    # Add a horizontal line at the average
    fig.add_hline(
        y=avg_sales, 
        line_dash='dash', 
        line_color='black', 
        opacity=0.5,
        annotation=dict(text='Average hourly sales', showarrow=False, xshift=-225, yshift=10)
    )

    # Update layout
    fig.update_layout(coloraxis_colorbar=dict(title='% vs. Average'))

    return hour_sales, fig

# Run the analysis
hour_data, hour_chart = analyze_hourly_sales(df)
hour_chart.show()

# Group by day period and calculate totals
period_data = hour_data.groupby('Day_Period').agg({
    'Total_Sales': 'sum',
    'Percentage': 'sum'
}).reset_index()

# Display the data tables
print("Hourly Sales Distribution:")
display(format_display_table(hour_data))

print("\nSales by Day Period:")
display(format_display_table(period_data))


# **Key Insight: Bimodal Shopping Pattern with Clear Peak Hours**
# Our hourly sales analysis reveals a distinctive bimodal distribution with two clear peak periods:
# 
# - **Primary Daytime Peak (10AM-3PM):** Generates \$1.2M-\$1.3M per hour, representing the strongest shopping window
# - **Secondary Evening Peak (8PM-10PM):** Shows renewed activity around $1.2M per hour
# - **Overnight Low (2AM-6AM):** The quietest period with minimal activity
# 
# This pattern aligns with typical online shopping behavior where customers browse during work breaks and after dinner. The bimodal distribution suggests our platform serves both professional customers shopping during business hours and evening shoppers who browse after their workday ends.
# These insights should inform:
# 
# - **Customer support hours:** Ensure strong coverage during 9AM-3PM and 7PM-10PM windows
# - **Website maintenance:** Schedule for 2AM-5AM to minimize disruption
# - **Flash sale timing:** Target 12PM or 8PM for maximum visibility
# - **Marketing automation:** Schedule email campaigns for 9AM or 7PM to catch customers before peak shopping hours
# 
# ## 3. Seasonal Analysis
# ### 3.1 Seasonal Sales Patterns

# In[9]:


def analyze_seasonal_sales(df):
    """
    Analyze seasonal sales patterns and category performance by season.

    Parameters:
        df (pd.DataFrame): The e-commerce dataset

    Returns:
        tuple: (season_total DataFrame, season_category DataFrame, plotly figure)
    """
    # Create a copy to avoid modifying the original
    df_seasonal = df.copy()

    # Create Quarter and Season
    if 'Quarter' not in df_seasonal.columns:
        df_seasonal['Quarter'] = df_seasonal['Order_Date'].dt.quarter

    # Map quarters to seasons
    season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    df_seasonal['Season'] = df_seasonal['Quarter'].map(season_map)

    # Calculate total sales by season
    season_total = df_seasonal.groupby('Season')['Total_Sales'].sum().reindex(season_order).reset_index()
    total_overall = season_total['Total_Sales'].sum()
    season_total['Percentage'] = (season_total['Total_Sales'] / total_overall * 100).round(1)

    # Calculate seasonal sales by category
    season_category = df_seasonal.groupby(['Season', 'Product_Category'])['Total_Sales'].sum().reset_index()

    # Create visualization
    fig = bar_chart(
        x='Season',
        y='Total_Sales',
        df=season_category,
        plot_title='Seasonal Sales by Product Category',
        x_label='Season',
        y_label='Total Revenue ($)',
        scale='millions',
        barmode='group',
        color='Product_Category',
        color_discrete_sequence=px.colors.qualitative.Plotly
    )

    # Add a second visualization for season contribution
    pie_fig = px.pie(
        season_total,
        values='Total_Sales',
        names='Season',
        title='Revenue Distribution by Season',
        color='Season',
        color_discrete_sequence=px.colors.qualitative.Plotly,
        hole=0.4
    )

    pie_fig.update_traces(
        textinfo='percent+label+value',
        texttemplate='%{label}<br>%{percent}<br>$%{value:,.0f}'
    )

    return season_total, season_category, fig, pie_fig

# Run the analysis
season_totals, season_categories, season_category_chart, season_pie_chart = analyze_seasonal_sales(df)

# Display the visualizations
season_category_chart.show()
season_pie_chart.show()

# Display the data tables
print("Sales by Season:")
display(format_display_table(season_totals))


# **Key Insight: Fall Dominance with Winter Weakness**
# Our seasonal analysis reveals a clear pattern with significant revenue variance across seasons:
# 
# - **Fall Dominance:** Fall generates \$6.04M (30.8% of annual revenue), making it our strongest season
# - **Summer and Spring Performance:** Summer (\$5.52M, 28.1%) and Spring (\$5.19M, 26.4%) deliver relatively balanced performance
# - **Winter Performance:** Winter significantly underperforms at \$2.88M (14.7%), about half the revenue of other seasons
# 
# This pattern occurs consistently across all product categories, with each showing its highest performance in Fall. The Winter weakness presents both a challenge and opportunity - it suggests a clear need for seasonal promotions, marketing campaigns, and potentially specialized products to boost this underperforming quarter.
# 
# The consistency of this pattern across categories indicates these are likely platform-wide trends related to shopping behavior rather than category-specific issues, possibly related to broader retail seasonality or our promotional calendar.
# 
# ### 3.2 Top Products by Season

# In[10]:


def analyze_seasonal_product_performance(df):
    """
    Identify and analyze top products for each season.

    Parameters:
        df (pd.DataFrame): The e-commerce dataset

    Returns:
        tuple: (DataFrame with top products for each season, plotly figure)
    """
    # Create a copy to avoid modifying the original
    df_seasonal = df.copy()

    # Ensure Season column exists
    if 'Season' not in df_seasonal.columns:
        df_seasonal['Quarter'] = df_seasonal['Order_Date'].dt.quarter
        season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
        df_seasonal['Season'] = df_seasonal['Quarter'].map(season_map)

    # Group by season and product
    season_products = df_seasonal.groupby(['Season', 'Product'])['Total_Sales'].sum().reset_index()

    # Create a list to store top products by season
    seasonal_top_products = []

    # Season order for display
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']

    # Find top products for each season
    for season in season_order:
        season_data = season_products[season_products['Season'] == season].sort_values('Total_Sales', ascending=False)
        top_3 = season_data.head(3).copy()  # Create explicit copy to avoid SettingWithCopyWarning

        # Calculate percentage of season revenue
        season_total = season_data['Total_Sales'].sum()
        top_3.loc[:, 'Percentage'] = (top_3['Total_Sales'] / season_total * 100).round(1)

        # Add rank and season (again) for easier filtering
        top_3.loc[:, 'Rank'] = range(1, len(top_3) + 1)

        # Add to list
        seasonal_top_products.append(top_3)

    # Combine results
    top_by_season = pd.concat(seasonal_top_products)

    # Create a more consistent rank label for display
    top_by_season['Rank_Label'] = top_by_season['Rank'].apply(lambda x: f'#{x}')

    fig = px.bar(
        top_by_season,
        x='Season',
        y='Total_Sales',
        color='Product',
        barmode='group',
        title='Top 3 Products by Season',
        labels={'Season': 'Season', 'Total_Sales': 'Revenue ($)'},
        template='plotly_white',
        text=top_by_season['Rank_Label'],
        category_orders={"Season": season_order}
    )

    # Update layout for better readability
    fig.update_traces(textposition='inside', textfont_color='white')
    fig.update_layout(
        legend_title='Product',
        height=500,
        bargap=0.2,
        bargroupgap=0.1
    )

    # Add annotations for rank labels
    fig.add_annotation(
        x=0.5, y=1.05,
        text="Note: Bar labels show product rank within each season (#1 = highest revenue)",
        showarrow=False,
        xref="paper", yref="paper",
        font=dict(size=12)
    )

    return top_by_season, fig

# Run the analysis
top_seasonal_products, seasonal_products_chart = analyze_seasonal_product_performance(df)
seasonal_products_chart.show()

# Display the data with custom formatting
for season in ['Winter', 'Spring', 'Summer', 'Fall']:
    season_data = top_seasonal_products[top_seasonal_products['Season'] == season]
    print(f"\nTop 3 Products in {season}:")
    for _, row in season_data.iterrows():
        print(f"  {row['Rank']}. {row['Product']}: ${row['Total_Sales']:,.2f} ({row['Percentage']}%)")


# **Key Insight: Seasonal Product Preferences Reveal Shifting Customer Demand**
# Our seasonal top-product analysis uncovers important shifts in customer preferences throughout the year:
# 
# 1. **T-Shirts Show Year-Round Appeal:** T-Shirts appear in the top 3 for all seasons but reach their peak in Fall (\$464K), suggesting they're a cornerstone product with year-round demand
# 2. **Watches and Shoes Demonstrate Seasonal Strength:**
# 
# - Titak Watch peaks in Spring (\$391K) and Fall (\$437K)
# - Running Shoes show strong Summer (\$402K) and Fall (\$411K) performance
# - Formal Shoes only reach the top 3 during Spring (\$358K)
# 
# 3. **Summer Shifts to Casual Wear:** Jeans peak in Summer ($423K), becoming the #1 product during warmer months
# 
# These seasonal shifts provide valuable insights for inventory planning and marketing strategy. For example:
# 
# - Winter promotions could focus on boosting T-Shirt and Watch sales, which remain strong despite overall seasonal weakness
# - Spring campaigns should highlight Formal Shoes, which uniquely reach their peak this season
# - Summer marketing should emphasize casual wear like Jeans and T-Shirts
# - Fall inventory should prioritize the "power trio" of T-Shirts, Watches, and Running Shoes
# 
# ## Interactive Dashboard

# In[11]:


def create_sales_dashboard(df):
    """
    Create an interactive dashboard for sales performance analysis.

    Parameters:
        df (pd.DataFrame): The e-commerce dataset

    Returns:
        widgets.VBox: A dashboard with interactive widgets
    """
    # Calculate necessary data
    total_revenue = df['Total_Sales'].sum()
    avg_monthly_revenue = df.groupby(df['Order_Date'].dt.to_period('M'))['Total_Sales'].sum().mean()
    num_transactions = len(df)
    avg_order_value = total_revenue / df['Order_Id'].nunique()

    # Create key metrics widgets
    metrics_html = f"""
    <div style="display:flex; justify-content:space-between; text-align:center; margin-bottom:20px;">
        <div style="background-color:#e6f7ff; padding:15px; border-radius:10px; width:24%;">
            <h3 style="margin:0; color:#0066cc;">Total Revenue</h3>
            <p style="font-size:24px; margin:10px 0; font-weight:bold;">${total_revenue:,.0f}</p>
        </div>
        <div style="background-color:#e6fff2; padding:15px; border-radius:10px; width:24%;">
            <h3 style="margin:0; color:#00994d;">Avg Monthly Revenue</h3>
            <p style="font-size:24px; margin:10px 0; font-weight:bold;">${avg_monthly_revenue:,.0f}</p>
        </div>
        <div style="background-color:#fff2e6; padding:15px; border-radius:10px; width:24%;">
            <h3 style="margin:0; color:#cc7a00;">Orders</h3>
            <p style="font-size:24px; margin:10px 0; font-weight:bold;">{num_transactions:,}</p>
        </div>
        <div style="background-color:#f2e6ff; padding:15px; border-radius:10px; width:24%;">
            <h3 style="margin:0; color:#7a00cc;">Avg Order Value</h3>
            <p style="font-size:24px; margin:10px 0; font-weight:bold;">${avg_order_value:,.2f}</p>
        </div>
    </div>
    """

    # Create time period selector
    time_periods = ['Monthly', 'Weekly', 'Daily', 'Hourly']
    time_dropdown = widgets.Dropdown(
        options=time_periods,
        value='Monthly',
        description='Time Period:',
        style={'description_width': 'initial'}
    )

    # Create category selector
    categories = ['All Categories'] + sorted(df['Product_Category'].unique().tolist())
    category_dropdown = widgets.Dropdown(
        options=categories,
        value='All Categories',
        description='Category:',
        style={'description_width': 'initial'}
    )

    # Function to update charts based on selections
    def update_charts(time_period, category):
        # Filter data if category is selected
        if category != 'All Categories':
            filtered_df = df[df['Product_Category'] == category]
        else:
            filtered_df = df

        # Create appropriate chart based on time period
        if time_period == 'Monthly':
            monthly_data, trend_fig, _ = analyze_monthly_sales(filtered_df)
            display(HTML("<h3>Monthly Sales Trend</h3>"))
            trend_fig.show()

        elif time_period == 'Weekly':
            weekly_data, weekly_fig = analyze_weekly_sales(filtered_df)
            display(HTML("<h3>Weekly Sales Trend</h3>"))
            weekly_fig.show()

        elif time_period == 'Daily':
            day_data, day_fig = analyze_day_of_week_sales(filtered_df)
            display(HTML("<h3>Daily Sales Trend</h3>"))
            day_fig.show()

        elif time_period == 'Hourly':
            hour_data, hour_fig = analyze_hourly_sales(filtered_df)
            display(HTML("<h3>Hourly Sales Trend</h3>"))
            hour_fig.show()

    # Create interactive output
    out = widgets.Output()

    # Function to handle interaction
    def on_change(change):
        out.clear_output()
        with out:
            update_charts(time_dropdown.value, category_dropdown.value)

    # Register callbacks
    time_dropdown.observe(on_change, names='value')
    category_dropdown.observe(on_change, names='value')

    # Create initial output
    with out:
        update_charts('Monthly', 'All Categories')

    # Create header
    header = widgets.HTML(
        value="<h1 style='text-align:center; color:#333;'>E-commerce Sales Performance Dashboard</h1>"
    )

    # Create metrics display
    metrics_display = widgets.HTML(value=metrics_html)

    # Create controls row
    controls = widgets.HBox([time_dropdown, category_dropdown])

    # Assemble the dashboard
    dashboard = widgets.VBox([
        header,
        metrics_display,
        widgets.HTML("<hr>"),
        widgets.HTML("<h2>Interactive Charts</h2>"),
        controls,
        out
    ])

    return dashboard

# Create and display the interactive dashboard
sales_dashboard = create_sales_dashboard(df)
display(sales_dashboard)


# ## Executive Summary and Recommendations

# In[12]:


def display_executive_summary():
    """Display an executive summary of the sales performance analysis"""

    summary_html = """
    <div style="background-color:#f8f9fa; padding:20px; border-radius:10px; margin-top:30px;">
        <h2 style="color:#333; border-bottom:2px solid #ddd; padding-bottom:10px;">Executive Summary: Sales Performance</h2>

        <h3 style="color:#0066cc;">Key Findings</h3>
        <ul>
            <li><strong>Fashion dominates revenue (57.5%)</strong> with T-Shirts as the top-selling product</li>
            <li><strong>Revenue concentration</strong> in top 10 products (58.7% of total revenue)</li>
            <li><strong>Distinct seasonal cycles</strong> with November peak (\$2.23M) and February trough (\$842K)</li>
            <li><strong>Weekday purchasing preference</strong> with Tuesday as the strongest day</li>
            <li><strong>Bimodal daily distribution</strong> with peaks at 10AM-3PM and 8PM-10PM</li>
            <li><strong>Fall generates highest sales</strong> (30.8%) while Winter underperforms (14.7%)</li>
        </ul>

        <h3 style="color:#00994d;">Strategic Recommendations</h3>
        <ol>
            <li><strong>Inventory Planning</strong>: Increase stock of top products (T-Shirts, Watches, Running Shoes) during peak seasons (Fall and late Spring)</li>
            <li><strong>Marketing Timing</strong>: Target campaigns for Tuesday mid-morning and evening hours when customer engagement is highest</li>
            <li><strong>Winter Growth Strategy</strong>: Develop specific campaigns and promotions to boost the underperforming winter season</li>
            <li><strong>Product Development</strong>: Consider expanding electronics offerings, which currently underperform relative to industry standards</li>
            <li><strong>Category Balance</strong>: While maintaining fashion focus, explore growth opportunities in Home & Furniture to reduce category concentration risk</li>
        </ol>

        <h3 style="color:#cc7a00;">Next Steps</h3>
        <p>This sales analysis provides a solid foundation for understanding our business performance. The next phase should include:</p>
        <ul>
            <li>Profitability analysis to identify most profitable products and categories</li>
            <li>Customer segmentation to understand purchasing behaviors across different customer groups</li>
            <li>Discount impact analysis to optimize promotional strategies</li>
            <li>Shipping cost analysis to identify optimization opportunities</li>
        </ul>
    </div>
    """

    return HTML(summary_html)

# Display the executive summary
display_executive_summary()


# In[ ]:





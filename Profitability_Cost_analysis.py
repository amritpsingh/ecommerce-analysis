#!/usr/bin/env python
# coding: utf-8

# # E-commerce Profitability & Cost Analysis
# ### Executive Summary
# This analysis examines the profitability drivers within our e-commerce platform, analyzing margins across product categories, the impact of discounts, and shipping costs. Our findings reveal:
# 
# - Fashion dominates both revenue (57.5%) and profit (59.9%) with consistent 45.5% profit margins
# - Premium-priced products generate disproportionate profits (62.1% of profits from 50.3% of revenue)
# - Higher discounts significantly impact margins - dropping from 49.1% at lowest discount to 41-42% at higher discount tiers
# - Electronics products have highest margins (59.1% for Apple Laptop) but low volume
# - Shipping costs impact high-ticket items most - reducing margins by 2-2.7% for premium electronics
# 
# The insights in this notebook will help optimize pricing strategies, discount structures, and product mix to maximize profitability while maintaining sales growth.
# Introduction
# While the previous analysis identified what sells and when, this notebook examines what drives profitability in our e-commerce business. We'll investigate:
# 
# 1. Which categories and products generate the highest profits and margins
# 2. How discounts affect profitability across different segments
# 3. The impact of shipping costs on overall margins
# 4. How price segments differ in their contribution to revenue and profit
# 
# This understanding will help optimize business decisions around pricing, promotions, and product strategy.
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


# In[2]:


# Load the data
df = load_data('ecommerce_cleaned.parquet')
df.head()


# ## 1. Profitability Analysis by Category

# In[3]:


def analyze_category_profitability(df):
    """
    Analyze profitability metrics by product category.

    Parameters:
        df (pd.DataFrame): The e-commerce dataset

    Returns:
        tuple: (category_metrics DataFrame, bar chart figure, comparison figure)
    """
    # Group by product category
    category_metrics = df.groupby('Product_Category', observed=True).agg({
        'Total_Sales': 'sum',
        'Total_Profit': 'sum',
        'Order_Id': 'nunique'  # Number of unique orders
    }).reset_index()

    # Calculate profit margin and contribution
    category_metrics['Profit_Margin'] = (category_metrics['Total_Profit'] / category_metrics['Total_Sales'] * 100).round(1)
    category_metrics['Revenue_Contribution'] = (category_metrics['Total_Sales'] / category_metrics['Total_Sales'].sum() * 100).round(1)
    category_metrics['Profit_Contribution'] = (category_metrics['Total_Profit'] / category_metrics['Total_Profit'].sum() * 100).round(1)

    # Sort by total profit
    category_metrics_sorted = category_metrics.sort_values('Total_Profit', ascending=False).reset_index(drop=True)

    # Create text labels for visualization
    category_metrics_sorted['Profit_Label'] = category_metrics_sorted['Total_Profit'].apply(
        lambda x: format_currency(x/1000000, 1) + 'M'
    )

    # Create a bar chart
    profit_chart = create_bar_chart(
        category_metrics_sorted,
        x='Product_Category',
        y='Total_Profit',
        title='Profit by Category with Margin %',
        x_label='Category',
        y_label='Total Profit ($)',
        color='Profit_Margin',
        text='Profit_Label',
        color_continuous_scale=px.colors.sequential.Viridis
    )

    # Create comparison chart
    comparison_chart = px.bar(
        category_metrics_sorted,
        x='Product_Category',
        y=['Revenue_Contribution', 'Profit_Contribution'],
        barmode='group',
        title='Revenue vs. Profit Contribution by Category',
        labels={
            'value': 'Percentage (%)', 
            'Product_Category': 'Category', 
            'variable': 'Metric'
        },
        color_discrete_map={
            'Revenue_Contribution': '#636EFA', 
            'Profit_Contribution': '#00CC96'
        }
    )

    # Update layout
    comparison_chart.update_layout(
        xaxis_title='Category',
        yaxis_title='Percentage (%)',
        xaxis_tickangle=-45,
        yaxis_ticksuffix='%'
    )

    return category_metrics_sorted, profit_chart, comparison_chart

# Run the analysis
category_data, category_profit_chart, category_comparison_chart = analyze_category_profitability(df)

# Display results
print("Profitability by Product Category:")
display(format_display_table(category_data))

# Show charts
category_profit_chart.show()
category_comparison_chart.show()


# **Key Insights on Category Profitability**
# The profitability analysis by category reveals important patterns that shape our business performance:
# 
# 1. **Fashion dominates both revenue (57.5%) and profit (59.9%)**, confirming its position as our core business. Its 45.5% profit margin is strong, making it both our largest and one of our most profitable categories.
# 2. **Electronics shows the highest margin (43.3%)** despite being our smallest category (4.4% of revenue). This suggests an opportunity to expand this high-margin category.
# 3. **Home & Furniture and Auto & Accessories** have slightly lower margins (41.0% and 41.1% respectively) but still contribute significantly to overall profit.
# 4. **Profit contribution exceeds revenue contribution for Fashion and Electronics,** indicating these categories are more efficient at converting sales into profit.
# 
# This analysis confirms that our business model is effectively optimized around our core Fashion category, but there could be growth opportunities in the high-margin Electronics segment.
# 
# ## 2. Product-Level Profitability Analysis

# In[4]:


def analyze_product_profitability(df):
    """
    Analyze profitability metrics at the product level.

    Parameters:
        df (pd.DataFrame): The e-commerce dataset

    Returns:
        tuple: (product_metrics DataFrame, top products DataFrame, margin products DataFrame, 
                bottom margin products DataFrame, profit bar chart, scatter plot)
    """
    # Group by product and category
    product_metrics = df.groupby(['Product', 'Product_Category'], observed=True).agg({
        'Total_Sales': 'sum',
        'Total_Profit': 'sum',
        'Unit_Price': 'first',
        'Quantity': 'sum'
    }).reset_index()

    # Calculate profit metrics
    product_metrics['Profit_Margin'] = (product_metrics['Total_Profit'] / product_metrics['Total_Sales'] * 100).round(1)
    product_metrics['Profit_per_Unit'] = (product_metrics['Total_Profit'] / product_metrics['Quantity']).round(2)

    # Get top products by different metrics
    top_profit_products = product_metrics.sort_values('Total_Profit', ascending=False).head(10)
    top_margin_products = product_metrics.sort_values('Profit_Margin', ascending=False).head(10)
    bottom_margin_products = product_metrics.sort_values('Profit_Margin', ascending=True).head(10)

    # Add profit labels for visualization
    top_profit_products['Profit_Label'] = top_profit_products['Total_Profit'].apply(
        lambda x: format_currency(x/1000, 1) + 'K'
    )

    # Create chart for top profitable products
    profit_chart = create_bar_chart(
        top_profit_products,
        x='Product',
        y='Total_Profit',
        title='Top 10 Products by Total Profit',
        x_label='Product',
        y_label='Total Profit ($)',
        color='Profit_Margin',
        text='Profit_Label',
        color_continuous_scale=px.colors.sequential.Viridis
    )

    # Create scatter plot of profit vs. revenue
    scatter_plot = create_scatter_plot(
        product_metrics.sort_values('Total_Sales', ascending=False).head(20),
        x='Total_Sales',
        y='Total_Profit',
        title='Profit vs. Revenue for Top 20 Products by Sales',
        x_label='Total Revenue ($)',
        y_label='Total Profit ($)',
        color='Profit_Margin',
        size='Quantity',
        hover_name='Product',
        color_continuous_scale=px.colors.sequential.Viridis
    )

    return (product_metrics, top_profit_products, top_margin_products, 
            bottom_margin_products, profit_chart, scatter_plot)

# Run the analysis
(product_data, top_profit_products, top_margin_products, 
 bottom_margin_products, product_profit_chart, product_scatter_plot) = analyze_product_profitability(df)

# Display results
print("Top 10 Most Profitable Products:")
display(format_display_table(top_profit_products))

print("\nTop 10 Products by Profit Margin:")
display(format_display_table(top_margin_products))

print("\nBottom 10 Products by Profit Margin:")
display(format_display_table(bottom_margin_products))

# Show charts
product_profit_chart.show()
product_scatter_plot.show()


# **Key Insights on Product Profitability**
# Our product-level profitability analysis reveals important patterns about which items drive business value:
# 
# 1. **T-Shirts are our profit powerhouse** - generating \$844K in profit with a strong 56.8% margin. This makes them both our highest total profit product and one of our highest margin items.
# 2. **Electronics lead in margin percentage** - Apple Laptop (59.1%), Samsung Mobile (54.3%), and Iron (54.3%) have the highest profit margins, but relatively low sales volume.
# 3. **Accessory items show impressive margins** - Products like Tyre (57.2%), Towels (54.4%), and Car Pillow & Neck Rest (54.2%) demonstrate that accessories can deliver premium margins.
# 4. **Fashion dominates the top 10 profit list** - Six of our top ten profit-generating products are Fashion items, confirming the category's importance to overall business success.
# 5. **Low-margin products are primarily basic items** - Watch (14.1%), Suits (15.6%), and Mouse (18.3%) have the lowest margins, suggesting they may be more commoditized or face stronger competition.
# 
# This analysis shows we should continue emphasizing our high-profit Fashion and accessory items while exploring opportunities to increase volume in our high-margin Electronics category.
# 
# ## 3. Top Products Within Each Category

# In[5]:


def analyze_category_top_products(product_metrics):
    """
    Identify and analyze the top products within each category.

    Parameters:
        product_metrics (pd.DataFrame): Product-level metrics

    Returns:
        tuple: (category_top_df DataFrame, bar chart figure)
    """
    # Create a list to store top products by category
    category_top_products = []

    # Find top products for each category
    for category in product_metrics['Product_Category'].unique():
        # Filter products by category
        category_products = product_metrics[product_metrics['Product_Category'] == category]
        # Get top 3 by profit
        top_3 = category_products.sort_values('Total_Profit', ascending=False).head(3).copy()

        # Add each of the top 3 products
        for i, (_, row) in enumerate(top_3.iterrows()):
            category_top_products.append({
                'Product_Category': category,
                'Product': row['Product'],
                'Total_Sales': row['Total_Sales'],
                'Total_Profit': row['Total_Profit'],
                'Profit_Margin': row['Profit_Margin'],
                'Rank': i + 1  # Proper ranking: 1, 2, 3
            })

    # Create DataFrame from the list
    category_top_df = pd.DataFrame(category_top_products)

    # Create rank label for visualization
    category_top_df['Rank_Label'] = category_top_df['Rank'].apply(lambda x: f'#{x}')

    # Create bar chart
    fig = px.bar(
        category_top_df,
        x='Product_Category',
        y='Total_Profit',
        color='Product',
        barmode='group',
        title='Top 3 Most Profitable Products by Category',
        labels={'Product_Category': 'Category', 'Total_Profit': 'Total Profit ($)'},
        text=category_top_df['Rank_Label']
    )

    # Update layout
    fig.update_traces(textposition='inside', textfont_color='white')
    fig.update_layout(
        xaxis_title='Category',
        yaxis_title='Total Profit ($)',
        legend_title='Product',
        height=500,
        bargap=0.2,
        bargroupgap=0.1,
        yaxis_tickprefix='$',
        yaxis_tickformat=','
    )

    # Add annotation explaining rank labels
    fig.add_annotation(
        x=0.5, y=1.05,
        text="Note: Bar labels show product rank within each category (#1 = highest profit)",
        showarrow=False,
        xref="paper", yref="paper",
        font=dict(size=12)
    )

    return category_top_df, fig

# Run the analysis
category_top_products_data, category_top_products_chart = analyze_category_top_products(product_data)

# Display results
print("Top 3 Most Profitable Products by Category:")
display(format_display_table(category_top_products_data))

# Show chart
category_top_products_chart.show()


# **Key Insights on Category Leaders**
# The analysis of top performers within each category reveals distinct product champions:
# 
# 1. **Fashion Category:** T-Shirts (\$844K profit, 56.8% margin) lead the category, followed by Titak Watch (\$767K) and Running Shoes (\$726K). All three maintain exceptional margins above 53%.
# 2. **Home & Furniture Category:** Towels (\$469K profit, 54.4% margin) are the standout performer, with Sofa Covers (\$435K) and Bed Sheets (\$412K) following closely. All maintain margins above 51%.
# 3. **Auto & Accessories Category:** Tyre (\$289K profit, 57.2% margin) leads this category, followed by Car Pillow & Neck Rest (\$252K) and Car Speakers (\$213K). These accessories demonstrate that auto products can achieve strong margins.
# 4. **Electronics Category:** Despite being our smallest category, Electronics features products with impressive margins - Apple Laptop (59.1%), Samsung Mobile (54.3%), and Iron (54.3%) all exceed 54% margin.
# 
# This analysis helps identify the "champions" within each category that should receive priority in inventory management, marketing focus, and merchandising placement. These products combine strong profitability with significant sales volume, making them ideal for driving business growth.
# 
# ## 4. Discount Impact Analysis

# In[6]:


def analyze_discount_impact(df):
    """
    Analyze the impact of discounts on sales and profitability.

    Parameters:
        df (pd.DataFrame): The e-commerce dataset

    Returns:
        tuple: (discount_impact DataFrame, margin bar chart, revenue bar chart, category line chart)
    """
    # Create a copy to avoid modifying the original
    df_discount = df.copy()

    # Create discount brackets
    df_discount['Discount_Bracket'] = pd.cut(
        df_discount['Discount'],
        bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0],
        labels=['0-10%', '11-20%', '21-30%', '31-40%', '41-50%', '51-100%'],
        include_lowest=True
    )

    # Aggregate metrics by discount bracket
    discount_impact = df_discount.groupby('Discount_Bracket', observed=True).agg({
        'Order_Id': 'nunique',    # Number of orders
        'Total_Sales': 'sum',
        'Total_Profit': 'sum',
        'Quantity': 'sum'         # Total quantity sold
    }).reset_index()

    # Calculate additional metrics
    discount_impact['Profit_Margin'] = (discount_impact['Total_Profit'] / discount_impact['Total_Sales'] * 100).round(1)
    discount_impact['Avg_Order_Value'] = (discount_impact['Total_Sales'] / discount_impact['Order_Id']).round(2)
    discount_impact['Orders_Percentage'] = (discount_impact['Order_Id'] / discount_impact['Order_Id'].sum() * 100).round(1)

    # Create margin labels for visualization
    discount_impact['Margin_Label'] = discount_impact['Profit_Margin'].apply(
        lambda x: f"{x:.1f}%" if not pd.isna(x) else "N/A"
    )

    # Create bar chart for profit margin by discount
    margin_chart = create_bar_chart(
        discount_impact,
        x='Discount_Bracket',
        y='Profit_Margin',
        title='Impact of Discounts on Profit Margin',
        x_label='Discount Level',
        y_label='Profit Margin (%)',
        color='Profit_Margin',
        text='Margin_Label',
        color_continuous_scale=px.colors.sequential.Viridis
    )

    # Create chart for revenue and profit by discount
    revenue_chart = px.bar(
        discount_impact,
        x='Discount_Bracket',
        y=['Total_Sales', 'Total_Profit'],
        barmode='group',
        title='Revenue and Profit by Discount Level',
        labels={'value': 'Amount ($)', 'Discount_Bracket': 'Discount Level', 'variable': 'Metric'}
    )

    revenue_chart.update_layout(
        xaxis_title='Discount Level',
        yaxis_title='Amount ($)',
        yaxis_tickprefix='$',
        yaxis_tickformat=','
    )

    # Analyze discount impact by category
    category_discount = df_discount.groupby(['Product_Category', 'Discount_Bracket'], observed=True).agg({
        'Total_Sales': 'sum',
        'Total_Profit': 'sum'
    }).reset_index()

    category_discount['Profit_Margin'] = (category_discount['Total_Profit'] / category_discount['Total_Sales'] * 100).round(1)

    # Create line chart for category margin by discount
    category_chart = create_line_chart(
        category_discount,
        x='Discount_Bracket',
        y='Profit_Margin',
        title='Profit Margin by Discount Level and Category',
        x_label='Discount Level',
        y_label='Profit Margin (%)',
        color='Product_Category',
        markers=True
    )

    return discount_impact, margin_chart, revenue_chart, category_chart

# Run the analysis
discount_data, discount_margin_chart, discount_revenue_chart, discount_category_chart = analyze_discount_impact(df)

# Display results
print("Discount Impact Analysis:")
display(format_display_table(discount_data))

# Show charts
discount_margin_chart.show()
discount_revenue_chart.show()
discount_category_chart.show()


# **Key Insights on Discount Impact**
# The discount analysis reveals critical patterns about how promotions affect profitability:
# 
# 1. **Clear profitability decline with higher discounts** - Profit margins drop significantly from 49.1% at the lowest discount tier (0-10%) to 41-42% at higher discount levels (21-50%).
# 2. **Optimal discount range appears to be 11-20%** - This tier balances healthy margins (45.3%) with strong sales volume, representing 23.9% of all orders.
# 3. **Higher discounts drive larger order values** - Average order value increases with discount level, from \$367.61 at 0-10% discount to \$410.70 at 41-50% discount, suggesting customers spend more when given deeper discounts.
# 4. **Highest volume occurs at 21-30% discount** - This tier accounts for 24.2% of all orders, indicating it may be the sweet spot for attracting customers.
# 5. **Electronics margins show highest discount sensitivity** - The steepest margin decline occurs in Electronics when moving from lower to higher discount tiers, while Fashion maintains more consistent margins across discount levels.
# 
# These insights suggest a tiered discount strategy might be optimal: limiting deep discounts on Electronics, potentially offering more aggressive promotions on Fashion items, and generally targeting the 11-20% range for the best balance of volume and profitability.
# 
# ## 5. Shipping Cost Analysis

# In[7]:


def analyze_shipping_costs(df):
    """
    Analyze the impact of shipping costs on profitability.

    Parameters:
        df (pd.DataFrame): The e-commerce dataset

    Returns:
        tuple: (shipping_by_category DataFrame, most_impacted DataFrame, 
                category bar chart, product bar chart)
    """
    # Create a copy to avoid modifying the original
    df_shipping = df.copy()

    # Calculate shipping as percentage of sales
    df_shipping['Shipping_Percentage'] = (df_shipping['Shipping_Cost'] / df_shipping['Total_Sales'] * 100).round(2)

    # Analyze by category
    shipping_by_category = df_shipping.groupby('Product_Category', observed=True).agg({
        'Total_Sales': 'sum',
        'Shipping_Cost': 'sum',
        'Total_Profit': 'sum'
    }).reset_index()

    # Calculate shipping metrics
    shipping_by_category['Shipping_Percentage'] = (shipping_by_category['Shipping_Cost'] / 
                                                 shipping_by_category['Total_Sales'] * 100).round(2)
    shipping_by_category['Profit_After_Shipping'] = shipping_by_category['Total_Profit'] - shipping_by_category['Shipping_Cost']
    shipping_by_category['Margin_After_Shipping'] = (shipping_by_category['Profit_After_Shipping'] / 
                                                   shipping_by_category['Total_Sales'] * 100).round(2)

    # Create percentage labels
    shipping_by_category['Percentage_Label'] = shipping_by_category['Shipping_Percentage'].apply(
        lambda x: f"{x:.2f}%"
    )

    # Create category chart
    category_chart = create_bar_chart(
        shipping_by_category,
        x='Product_Category',
        y='Shipping_Percentage',
        title='Shipping Cost as Percentage of Sales by Category',
        x_label='Category',
        y_label='Shipping Cost (% of Sales)',
        color='Shipping_Percentage',
        text='Percentage_Label',
        color_continuous_scale=px.colors.sequential.Blues_r
    )

    # Analyze by product
    product_shipping = df_shipping.groupby('Product', observed=True).agg({
        'Total_Sales': 'sum',
        'Shipping_Cost': 'sum',
        'Total_Profit': 'sum'
    }).reset_index()

    # Calculate product shipping metrics
    product_shipping['Shipping_Percentage'] = (product_shipping['Shipping_Cost'] / 
                                              product_shipping['Total_Sales'] * 100).round(2)
    product_shipping['Profit_Before_Shipping'] = product_shipping['Total_Profit'] + product_shipping['Shipping_Cost']
    product_shipping['Margin_Before_Shipping'] = (product_shipping['Profit_Before_Shipping'] / 
                                                product_shipping['Total_Sales'] * 100).round(2)
    product_shipping['Margin_After_Shipping'] = (product_shipping['Total_Profit'] / 
                                               product_shipping['Total_Sales'] * 100).round(2)
    product_shipping['Margin_Impact'] = product_shipping['Margin_Before_Shipping'] - product_shipping['Margin_After_Shipping']

    # Get products most impacted by shipping
    most_impacted = product_shipping.sort_values('Margin_Impact', ascending=False).head(10)

    # Create product chart
    product_chart = px.bar(
        most_impacted,
        x='Product',
        y=['Margin_Before_Shipping', 'Margin_After_Shipping'],
        barmode='group',
        title='Impact of Shipping on Profit Margins (Top 10 Most Affected Products)',
        labels={'value': 'Profit Margin (%)', 'Product': 'Product', 'variable': 'Margin Type'}
    )

    product_chart.update_layout(
        xaxis_title='Product',
        yaxis_title='Profit Margin (%)',
        xaxis_tickangle=-45,
        yaxis_ticksuffix='%'
    )

    return shipping_by_category, most_impacted, category_chart, product_chart

# Run the analysis
shipping_category_data, shipping_product_data, shipping_category_chart, shipping_product_chart = analyze_shipping_costs(df)

# Display results
print("Shipping Cost Analysis by Category:")
display(format_display_table(shipping_category_data))

print("\nTop 10 Products Most Impacted by Shipping Costs:")
display(format_display_table(shipping_product_data))

# Show charts
shipping_category_chart.show()
shipping_product_chart.show()


# **Key Insights on Shipping Costs**
# The shipping cost analysis reveals important patterns about how fulfillment expenses impact profitability:
# 
# 1. **Electronics face highest shipping impact** - Electronics has the highest shipping cost as a percentage of sales (2.01%), followed by Auto & Accessories (1.88%), which slightly reduces their otherwise strong margins.
# 2. **Premium products show greatest margin reduction** - High-value items like Apple Laptop (-2.69 percentage points), Tyre (-2.62), and Iron (-2.58) see the largest margin impact from shipping costs.
# 3. **Fashion has lowest shipping burden** - The Fashion category has the lowest shipping percentage (1.83%), helping maintain its strong overall profitability.
# 4. **Shipping impact is generally manageable** - Across all categories, shipping costs represent less than 2.1% of sales, suggesting our current shipping strategy is relatively efficient.
# 5. **Shipping affects margin more than volume** - There's no evidence that shipping costs significantly impact purchase volume; they primarily affect profit margins.
# 
# These insights suggest targeting shipping cost optimization for Electronics and high-value products could yield meaningful margin improvements. Consolidating orders (especially for multiple small Electronics items) or offering free shipping with minimum purchase thresholds could help mitigate these impacts.
# 
# ## 6. Value Segment Analysis

# In[8]:


def analyze_value_segments(df):
    """
    Analyze performance metrics across different price segments.

    Parameters:
        df (pd.DataFrame): The e-commerce dataset

    Returns:
        tuple: (segment_analysis DataFrame, contribution bar chart, discount line chart)
    """
    # Create a copy to avoid modifying the original
    df_segment = df.copy()

    # Create price segments based on unit price distribution
    price_quantiles = df_segment['Unit_Price'].quantile([0.33, 0.66]).values
    df_segment['Price_Segment'] = pd.cut(
        df_segment['Unit_Price'],
        bins=[0, price_quantiles[0], price_quantiles[1], df_segment['Unit_Price'].max()],
        labels=['Budget', 'Mid-range', 'Premium'],
        include_lowest=True
    )

    # Analyze metrics by price segment
    segment_analysis = df_segment.groupby('Price_Segment', observed=True).agg({
        'Order_Id': 'nunique',    # Number of orders
        'Total_Sales': 'sum',
        'Total_Profit': 'sum',
        'Discount': 'mean',       # Average discount
        'Shipping_Cost': 'sum'    # Total shipping cost
    }).reset_index()

    # Calculate additional metrics
    segment_analysis['Profit_Margin'] = (segment_analysis['Total_Profit'] / 
                                       segment_analysis['Total_Sales'] * 100).round(1)
    segment_analysis['Revenue_Contribution'] = (segment_analysis['Total_Sales'] / 
                                              segment_analysis['Total_Sales'].sum() * 100).round(1)
    segment_analysis['Profit_Contribution'] = (segment_analysis['Total_Profit'] / 
                                             segment_analysis['Total_Profit'].sum() * 100).round(1)
    segment_analysis['Avg_Discount'] = (segment_analysis['Discount'] * 100).round(1)

    # Create contribution chart - Using px.bar directly instead of create_bar_chart
    contribution_chart = px.bar(
        segment_analysis,
        x='Price_Segment',
        y=['Revenue_Contribution', 'Profit_Contribution'],
        barmode='group',
        title='Revenue vs. Profit Contribution by Price Segment',
        labels={'value': 'Percentage (%)', 'Price_Segment': 'Price Segment', 'variable': 'Metric'}
    )

    # Update layout
    contribution_chart.update_layout(
        xaxis_title='Price Segment',
        yaxis_title='Contribution (%)',
        yaxis_ticksuffix='%'
    )

    # Analyze discount effectiveness by segment
    # Create discount brackets on the copy to avoid affecting original
    df_segment['Discount_Bracket'] = pd.cut(
        df_segment['Discount'],
        bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0],
        labels=['0-10%', '11-20%', '21-30%', '31-40%', '41-50%', '51-100%'],
        include_lowest=True
    )

    segment_discount = df_segment.groupby(['Price_Segment', 'Discount_Bracket'], observed=True).agg({
        'Total_Sales': 'sum',
        'Total_Profit': 'sum',
        'Order_Id': 'nunique'
    }).reset_index()

    segment_discount['Profit_Margin'] = (segment_discount['Total_Profit'] / 
                                       segment_discount['Total_Sales'] * 100).round(1)
    segment_discount['Avg_Order_Value'] = (segment_discount['Total_Sales'] / 
                                         segment_discount['Order_Id']).round(2)

    # Create discount impact chart
    discount_chart = create_line_chart(
        segment_discount,
        x='Discount_Bracket',
        y='Profit_Margin',
        title='Discount Impact on Profit Margin by Price Segment',
        x_label='Discount Level',
        y_label='Profit Margin (%)',
        color='Price_Segment',
        markers=True
    )

    return segment_analysis, contribution_chart, discount_chart

# Run the analysis
segment_data, segment_contribution_chart, segment_discount_chart = analyze_value_segments(df)

# Display results
print("Performance by Price Segment:")
display(format_display_table(segment_data))

# Show charts
segment_contribution_chart.show()
segment_discount_chart.show()


# **Key Insights on Price Segments**
# The price segment analysis reveals crucial patterns about how different product tiers contribute to business performance:
# 
# 1. **Premium products drive disproportionate profits** - Premium-priced products generate 62.1% of total profits from 50.3% of revenue, demonstrating exceptional efficiency at converting sales to profit.
# 2. **Dramatic profit margin differences across segments** - Premium products achieve 53.9% margins compared to just 25.9% for Budget items, highlighting the value of focusing on higher-priced merchandise.
# 3. **Budget items underperform significantly** - Budget products generate only 10.2% of profits despite representing 17.2% of revenue, making them our least efficient segment.
# 4. **Mid-range products maintain healthy balance** - With 37.3% margins, mid-range products deliver solid profitability while appealing to value-conscious customers.
# 5. **Premium products show higher discount rates** - Interestingly, Premium products receive slightly higher average discounts (32.0%) compared to Budget products (29.4%), suggesting we may be discounting our most profitable items too aggressively.
# 6. **Premium margins remain superior even with discounts** - Even at higher discount levels, Premium products maintain significantly better margins than Budget items at any discount tier.
# 
# These insights suggest focusing more resources on promoting Premium products, potentially reducing discounts on this highly profitable segment, and reconsidering the role of Budget items in our product mix.
# 
# ## 7. Executive Dashboard

# In[9]:


def create_profitability_dashboard(df, category_metrics, product_metrics, discount_impact, segment_analysis):
    """
    Create a comprehensive dashboard summarizing key profitability metrics.

    Parameters:
        df (pd.DataFrame): The e-commerce dataset
        category_metrics (pd.DataFrame): Category-level profitability
        product_metrics (pd.DataFrame): Product-level profitability
        discount_impact (pd.DataFrame): Discount impact analysis
        segment_analysis (pd.DataFrame): Price segment analysis

    Returns:
        plotly.graph_objects.Figure: The dashboard figure
    """
    # Calculate key overall metrics
    total_revenue = df['Total_Sales'].sum()
    total_profit = df['Total_Profit'].sum()
    overall_margin = (total_profit / total_revenue * 100).round(1)
    total_orders = df['Order_Id'].nunique()
    avg_order_value = (total_revenue / total_orders).round(2)

    # Sort category metrics by profit
    category_metrics_sorted = category_metrics.sort_values('Total_Profit', ascending=False)

    # Get top 5 products by profit
    top_5_profit = product_metrics.sort_values('Total_Profit', ascending=False).head(5)

    # Calculate monthly profit
    monthly_profit = df.groupby(df['Order_Date'].dt.to_period('M')).agg({
        'Total_Profit': 'sum'
    }).reset_index()
    monthly_profit['Month'] = monthly_profit['Order_Date'].dt.strftime('%b')
    monthly_profit['Month_Num'] = monthly_profit['Order_Date'].dt.month
    monthly_profit = monthly_profit.sort_values('Month_Num')

    # Calculate shipping costs by category
    shipping_by_category = df.groupby('Product_Category', observed=True).agg({
        'Total_Sales': 'sum',
        'Shipping_Cost': 'sum'
    }).reset_index()
    shipping_by_category['Shipping_Percentage'] = (shipping_by_category['Shipping_Cost'] / 
                                                 shipping_by_category['Total_Sales'] * 100).round(2)

    # Create dashboard layout
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Profit by Category', 
            'Top 5 Products by Profit',
            'Profit Margin by Discount Level',
            'Revenue vs. Profit by Price Segment',
            'Monthly Profit Trends',
            'Shipping Cost Impact by Category'
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "bar"}]
        ]
    )

    # 1. Category profit
    for i, row in category_metrics_sorted.iterrows():
        fig.add_trace(
            go.Bar(
                x=[row['Product_Category']], 
                y=[row['Total_Profit']],
                text=[f"${row['Total_Profit']/1000000:.1f}M<br>{row['Profit_Margin']}%"],
                name=row['Product_Category'],
                textposition='outside'
            ),
            row=1, col=1
        )

    # 2. Top 5 products by profit
    fig.add_trace(
        go.Bar(
            x=top_5_profit['Product'],
            y=top_5_profit['Total_Profit'],
            text=[f"${p/1000:.0f}K" for p in top_5_profit['Total_Profit']],
            textposition='outside'
        ),
        row=1, col=2
    )

    # 3. Profit margin by discount
    fig.add_trace(
        go.Bar(
            x=discount_impact['Discount_Bracket'],
            y=discount_impact['Profit_Margin'],
            text=[f"{m:.1f}%" for m in discount_impact['Profit_Margin'] if not pd.isna(m)],
            textposition='outside'
        ),
        row=2, col=1
    )

    # 4. Revenue vs Profit by segment
    fig.add_trace(
        go.Bar(
            x=segment_analysis['Price_Segment'],
            y=segment_analysis['Revenue_Contribution'],
            name='Revenue %',
            text=[f"{c:.1f}%" for c in segment_analysis['Revenue_Contribution']],
            textposition='outside'
        ),
        row=2, col=2
    )
    fig.add_trace(
        go.Bar(
            x=segment_analysis['Price_Segment'],
            y=segment_analysis['Profit_Contribution'],
            name='Profit %',
            text=[f"{c:.1f}%" for c in segment_analysis['Profit_Contribution']],
            textposition='outside'
        ),
        row=2, col=2
    )

    # 5. Monthly profit trends
    fig.add_trace(
        go.Scatter(
            x=monthly_profit['Month'],
            y=monthly_profit['Total_Profit'],
            mode='lines+markers+text',
            text=[f"${p/1000:.0f}K" for p in monthly_profit['Total_Profit']],
            textposition='top center'
        ),
        row=3, col=1
    )

    # 6. Shipping cost impact
    fig.add_trace(
        go.Bar(
            x=shipping_by_category['Product_Category'],
            y=shipping_by_category['Shipping_Percentage'],
            text=[f"{p:.2f}%" for p in shipping_by_category['Shipping_Percentage']],
            textposition='outside'
        ),
        row=3, col=2
    )

    # Update axis labels
    fig.update_xaxes(title_text="Category", row=1, col=1)
    fig.update_yaxes(title_text="Profit ($)", row=1, col=1)

    fig.update_xaxes(title_text="Product", row=1, col=2)
    fig.update_yaxes(title_text="Profit ($)", row=1, col=2)

    fig.update_xaxes(title_text="Discount Level", row=2, col=1)
    fig.update_yaxes(title_text="Profit Margin (%)", row=2, col=1)

    fig.update_xaxes(title_text="Price Segment", row=2, col=2)
    fig.update_yaxes(title_text="Contribution (%)", row=2, col=2)

    fig.update_xaxes(title_text="Month", row=3, col=1)
    fig.update_yaxes(title_text="Profit ($)", row=3, col=1)

    fig.update_xaxes(title_text="Category", row=3, col=2)
    fig.update_yaxes(title_text="Shipping Cost (%)", row=3, col=2)

    # Update layout
    fig.update_layout(
        title_text=f"Profitability Dashboard (Overall Margin: {overall_margin}%)",
        height=900,
        width=1200,
        showlegend=False
    )

    return fig

# Create the executive dashboard
dashboard = create_profitability_dashboard(
    df, category_data, product_data, discount_data, segment_data
)

# Show the dashboard
dashboard.show()


# ## 8. Strategic Recommendations

# In[10]:


def display_strategic_recommendations():
    """
    Display strategic recommendations based on profitability analysis.
    """
    recommendations_html = """
    <div style="background-color:#f8f9fa; padding:20px; border-radius:10px; margin-top:30px;">
        <h2 style="color:#333; border-bottom:2px solid #ddd; padding-bottom:10px;">Strategic Recommendations</h2>

        <div style="display:flex; flex-wrap:wrap; justify-content:space-between;">
            <div style="width:48%; background-color:#e8f4f8; padding:15px; border-radius:8px; margin-bottom:15px;">
                <h3 style="color:#0066cc;">1. Product Mix Optimization</h3>
                <p>Focus on expanding our Premium segment offerings, which generate 62.1% of profits from 50.3% of revenue. Consider culling underperforming Budget items that contribute only 10.2% of profits despite 17.2% of revenue.</p>
                <p><strong>Action Steps:</strong> Increase inventory and marketing for top margin products (Apple Laptop, Tyre, T-Shirts); review Budget category for potential discontinuation of lowest margin items.</p>
            </div>

            <div style="width:48%; background-color:#e8f8f4; padding:15px; border-radius:8px; margin-bottom:15px;">
                <h3 style="color:#00994d;">2. Discount Strategy Refinement</h3>
                <p>Optimize discount tiers based on margin impact - our current 11-20% discount level provides the best balance of sales volume and profitability. Reduce discounting of Premium products, which remain profitable even at lower discount levels.</p>
                <p><strong>Action Steps:</strong> Implement category-specific discount caps (max 20% for Electronics, 30% for Fashion); test discount removal on top 5 profit products to measure price elasticity.</p>
            </div>

            <div style="width:48%; background-color:#f8f4e8; padding:15px; border-radius:8px; margin-bottom:15px;">
                <h3 style="color:#cc7a00;">3. Category Growth Initiatives</h3>
                <p>Expand our high-margin Electronics category (43.3% margin) which currently contributes only 4.4% of revenue. Consider launching a targeted Electronics expansion with curated high-margin products.</p>
                <p><strong>Action Steps:</strong> Introduce 3-5 new premium Electronics products per quarter; develop Electronics-specific marketing campaign highlighting quality and features rather than discounts.</p>
            </div>

            <div style="width:48%; background-color:#f4e8f8; padding:15px; border-radius:8px; margin-bottom:15px;">
                <h3 style="color:#7a00cc;">4. Shipping Optimization</h3>
                <p>Address shipping cost impact on high-value products like Apple Laptop (-2.69 percentage points on margin). Develop targeted shipping promotions and bundling strategies to maintain margins while meeting customer expectations.</p>
                <p><strong>Action Steps:</strong> Introduce free shipping thresholds specific to each category; test bundled shipping rates for multiple items; optimize fulfillment locations for high-volume products.</p>
            </div>
        </div>

        <h3 style="color:#333; margin-top:20px;">Implementation Timeline:</h3>
        <ol>
            <li><strong>Immediate (1-30 days):</strong> Adjust discount tiers based on profitability analysis</li>
            <li><strong>Short-term (30-90 days):</strong> Review and optimize product mix, especially Budget category items</li>
            <li><strong>Medium-term (3-6 months):</strong> Implement shipping optimization strategies</li>
            <li><strong>Long-term (6-12 months):</strong> Execute Electronics category expansion plan</li>
        </ol>

        <p style="font-style:italic; margin-top:20px;">These recommendations should be implemented with continuous monitoring of key performance indicators including overall profit margin, category contribution, and discount effectiveness.</p>
    </div>
    """

    return HTML(recommendations_html)

# Display the strategic recommendations
display_strategic_recommendations()


# In[ ]:





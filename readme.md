# E-commerce Analysis Project

This repository contains a comprehensive analysis of an e-commerce platform's performance across three key dimensions:

1. **Sales Performance Analysis** - Understanding what products sell and when
2. **Profitability & Cost Analysis** - Examining margins and cost factors
3. **Customer Segmentation Analysis** - Identifying and analyzing customer segments

## Project Structure

- **Data Cleaning Notebooks**:
  - `data_cleaning.ipynb` - Performs data cleaning and transformation steps

- **Analysis Notebooks**:
  - `Sales_performance_analysis.ipynb` - Analyzes product sales and temporal patterns
  - `Profitability_cost_analysis.ipynb` - Examines profitability drivers and cost impacts
  - `Customer_segmentation_analysis.ipynb` - Segments customers and analyzes behavior

- **Utilities**:
  - `ecommerce_utils.py` - Common utility functions for data loading, formatting, and visualization

- **Setup**:
  - `requirements.txt` - Python package dependencies

## Key Insights

### Sales Performance Analysis
- Fashion dominates revenue (57.5%) with T-Shirts as the top-selling product
- Strong seasonality with Fall generating 30.8% of annual revenue
- High sales concentration with the top 10 products contributing 58.7% of total revenue

### Profitability & Cost Analysis
- Premium-priced products generate disproportionate profits (62.1% of profits from 50.3% of revenue)
- Higher discounts significantly impact margins - dropping from 49.1% at lowest discount to 41-42% at higher discount tiers
- Shipping costs impact high-ticket items most - reducing margins by 2-2.7% for premium electronics

### Customer Segmentation Analysis
- 73.9% of customers are one-time purchasers yet generate 55.9% of revenue
- "Champions" (our high-value customers) represent 22.9% of customers but drive 43.1% of revenue
- Fashion category preference varies by segment - Champions prefer Fashion (58-60%) while At-Risk customers favor Home & Furniture (56.9%)

## Setup Instructions

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/ecommerce-analysis.git
   cd ecommerce-analysis
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

4. Prepare your data:
   - Place your `ecommerce_cleaned.parquet` file in the project directory
   - Ensure the file contains the necessary columns (review the notebooks for required fields)


## License

This project is licensed under the MIT License - see the LICENSE file for details.

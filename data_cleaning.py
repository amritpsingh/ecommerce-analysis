import pandas as pd
import numpy as np
from datetime import datetime

def load_data(file_path):
    """Load e-commerce dataset from CSV file"""
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def convert_data_types(df):
    """Convert columns to appropriate data types"""
    try:
        # Create a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Convert date columns to datetime
        try:
            df_clean['Order_Date'] = pd.to_datetime(df_clean['Order_Date'])
            df_clean['Order_DateTime'] = pd.to_datetime(df_clean['Order_Date'].astype(str) + ' ' + df_clean['Time'])
        except Exception as e:
            print(f"Warning: Date conversion error: {e}. Using alternative format.")
            # Try alternative format if standard format fails
            df_clean['Order_Date'] = pd.to_datetime(df_clean['Order_Date'], format='%d-%m-%Y')
            df_clean['Order_DateTime'] = pd.to_datetime(df_clean['Order_Date'].astype(str) + ' ' + df_clean['Time'])
        
        # Extract time components
        df_clean['Year'] = df_clean['Order_Date'].dt.year
        df_clean['Month'] = df_clean['Order_Date'].dt.month
        df_clean['Day'] = df_clean['Order_Date'].dt.day
        df_clean['Week'] = df_clean['Order_Date'].dt.isocalendar().week
        df_clean['Weekday'] = df_clean['Order_Date'].dt.day_name()
        df_clean['Hour'] = df_clean['Order_DateTime'].dt.hour
        
        # Ensure numerical columns have correct types
        numerical_columns = ['Customer_Id', 'Aging', 'Sales', 'Quantity', 'Discount', 'Profit', 'Shipping_Cost']
        for column in numerical_columns:
            if column in df_clean.columns:
                df_clean[column] = pd.to_numeric(df_clean[column], errors='coerce')
        
        print("Data types successfully converted")
        return df_clean
    except Exception as e:
        print(f"Error in data type conversion: {e}")
        raise

def clean_text_fields(df):
    """Standardize text fields for consistency"""
    try:
        # Identify text columns
        text_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove Time and Order_Date if they exist in text_columns
        if 'Time' in text_columns:
            text_columns.remove('Time')
        if 'Order_Date' in text_columns:
            text_columns.remove('Order_Date')
        
        # Standardize each text column
        for column in text_columns:
            # Check if column exists and contains string data
            if column in df.columns and df[column].dtype == 'object':
                # Strip whitespace
                df[column] = df[column].str.strip()
                # Standardize capitalization (Title Case)
                df[column] = df[column].str.title()
        
        print(f"Standardized {len(text_columns)} text fields")
        return df
    except Exception as e:
        print(f"Error cleaning text fields: {e}")
        raise


def handle_missing_values(df):
    """Handle all missing values in the dataset with appropriate methods"""
    print("Handling missing values...")
    
    # 1. Handle missing Discount values
    for idx, row in df[df['Discount'].isna()].iterrows():
        product = row['Product']
        week = row['Week']
        year = row['Year']
        
        # Get discount for the same product in the same week
        same_week_discount = df[
            (df['Product'] == product) &
            (df['Week'] == week) &
            (df['Year'] == year) &
            (df['Discount'].notna())
        ]['Discount']
        
        if len(same_week_discount) > 0:
            # Use median of same week discounts
            discount_fill_value = same_week_discount.median()
        else:
            # If no data for same week, then may be there is no discount for the product that week
            discount_fill_value = 0
                
        df.loc[idx, 'Discount'] = discount_fill_value
        print(f"Filled missing discount for {product} (Week {week}) with: {discount_fill_value}")
    
    # 2. Handle missing Quantity values
    missing_quantity_products = df[df['Quantity'].isna()]['Product'].unique()
    for product in missing_quantity_products:
        # Calculate typical quantity for this product
        product_quantities = df[(df['Product'] == product) & (df['Quantity'].notna())]['Quantity']
        if len(product_quantities) > 0:
            frequent_quantity = product_quantities.mode()
            if len(frequent_quantity) > 0:
                fill_quantity = frequent_quantity.iloc[0]
            else:
                fill_quantity = product_quantities.median()
        else:
            # If no data for this product, use global mode
            fill_quantity = df['Quantity'].mode().iloc[0]
            
        df.loc[(df['Quantity'].isna()) & (df['Product'] == product), 'Quantity'] = fill_quantity
        print(f"Filled missing quantity for {product} with: {fill_quantity}")
    
    # 3. Handle missing Sales values
    for idx, row in df[df['Sales'].isna()].iterrows():
        product = row['Product']
        sales_values = df[(df['Product'] == product) & (df['Sales'].notna())]['Sales']
        
        if len(sales_values) > 0:
            sales_fill_value = sales_values.median()
        else:
            # If no data for this product, use category median
            category = row['Product_Category']
            sales_fill_value = df[(df['Product_Category'] == category) & (df['Sales'].notna())]['Sales'].median()
            
        df.loc[idx, 'Sales'] = sales_fill_value
        print(f"Filled missing sales for {product} with: {sales_fill_value}")
    
    # 4. Handle missing Shipping Cost
    for idx, row in df[df['Shipping_Cost'].isna()].iterrows():
        product = row['Product']
        shipping_values = df[(df['Product'] == product) & (df['Shipping_Cost'].notna())]['Shipping_Cost']
        
        if len(shipping_values) > 0:
            shipping_fill_value = shipping_values.median()
        else:
            # If no data for this product, use category median
            category = row['Product_Category']
            shipping_fill_value = df[(df['Product_Category'] == category) & 
                                    (df['Shipping_Cost'].notna())]['Shipping_Cost'].median()
            
        df.loc[idx, 'Shipping_Cost'] = shipping_fill_value
        print(f"Filled missing shipping cost for {product} with: {shipping_fill_value}")
    
    # 5. Handle missing Aging
    for idx, row in df[df['Aging'].isna()].iterrows():
        priority = row['Order_Priority']
        aging_values = df[(df['Order_Priority'] == priority) & (df['Aging'].notna())]['Aging']
        
        if len(aging_values) > 0:
            aging_fill_value = aging_values.median()
        else:
            # If no data for this priority, use global median
            aging_fill_value = df['Aging'].median()
            
        df.loc[idx, 'Aging'] = aging_fill_value
        print(f"Filled missing aging for priority '{priority}' with: {aging_fill_value}")
    
    # 6. Handle missing Order_Priority
    df['Order_Priority'].fillna('Not Specified', inplace=True)
    print("Filled missing Order_Priority with 'Not Specified'")
    
    return df


def create_business_metrics(df):
    """Create enhanced business metrics for analysis"""
    try:
        # Create unique order identifiers
        df['Order_Id'] = (
            df['Customer_Id'].astype(str) + '_' +
            df['Order_Date'].dt.strftime('%Y-%m-%d') + '_' +
            df['Time']
        )
        
        # Create financial metrics
        df['Total_Sales'] = df['Sales'] * df['Quantity']
        df['Total_Profit'] = df['Profit'] * df['Quantity']
        df['Profit_Margin'] = (df['Total_Profit'] / df['Total_Sales'] * 100).round(2)
        df['Discount_Amount'] = df['Total_Sales'] * df['Discount']
        df['Net_Sales'] = df['Total_Sales'] - df['Discount_Amount']
        df['Average_Item_Value'] = df['Total_Sales'] / df['Quantity']
        
        # Customer value metrics
        df['Revenue_After_Shipping'] = df['Net_Sales'] - df['Shipping_Cost']
        df['Net_Profit_Margin'] = (df['Total_Profit'] / df['Net_Sales'] * 100).round(2)
        
        # Rename original columns for clarity
        df.rename(columns={'Sales': 'Unit_Price', 'Profit': 'Unit_Profit'}, inplace=True)
        
        print("Enhanced business metrics created successfully")
        return df
    except Exception as e:
        print(f"Error creating business metrics: {e}")
        raise

def validate_data(df):
    """Perform validation checks on the cleaned data"""
    try:
        validation_results = []
        
        # Check 1: No negative Quantities
        neg_quantity = df[df['Quantity'] < 0]
        validation_results.append(("Negative Quantities", len(neg_quantity)))
        
        # Check 2: No negative Prices
        neg_price = df[df['Unit_Price'] < 0]
        validation_results.append(("Negative Prices", len(neg_price)))
        
        # Check 3: Discount range validation (0-100%)
        invalid_discount = df[(df['Discount'] < 0) | (df['Discount'] > 1)]
        validation_results.append(("Invalid Discounts", len(invalid_discount)))
        
        # Check 4: Order_ID uniqueness
        duplicate_orders = df['Order_Id'].duplicated().sum()
        validation_results.append(("Duplicate Order IDs", duplicate_orders))
        
        # Check 5: Date range consistency
        date_range = f"{df['Order_Date'].min().date()} to {df['Order_Date'].max().date()}"
        validation_results.append(("Date Range", date_range))
        
        # Display validation results
        validation_df = pd.DataFrame(validation_results, columns=['Check', 'Result'])
        print("\nData Validation Results:")
        print(validation_df)
        
        # Flag if any validation checks failed
        validation_failed = any(
            count > 0 for check, count in validation_results 
            if isinstance(count, int) and 'Date Range' not in check
        )
        
        if validation_failed:
            print("WARNING: Some validation checks failed!")
        else:
            print("All validation checks passed!")
            
        return validation_df, validation_failed
    except Exception as e:
        print(f"Error during data validation: {e}")
        raise

def create_summary_stats(df):
    """Generate summary statistics for the cleaned dataset"""
    try:
        summary_stats = {
            'Total Records': len(df),
            'Unique Customers': df['Customer_Id'].nunique(),
            'Unique Products': df['Product'].nunique(),
            'Product Categories': df['Product_Category'].nunique(),
            'Date Range': f"{df['Order_Date'].min().date()} to {df['Order_Date'].max().date()}",
            'Total Revenue': f"${df['Total_Sales'].sum():,.2f}",
            'Total Profit': f"${df['Total_Profit'].sum():,.2f}",
            'Average Order Value': f"${df['Total_Sales'].mean():,.2f}",
            'Average Profit Margin': f"{df['Profit_Margin'].mean():.1f}%"
        }
        
        summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
        print("\nCleaned Dataset Summary:")
        print(summary_df)
        
        return summary_df
    except Exception as e:
        print(f"Error generating summary statistics: {e}")
        raise

def clean_ecommerce_data(file_path, output_path=None, file_format='parquet'):
    """
    Main function to clean and transform e-commerce dataset
    
    Parameters:
    -----------
    file_path : str
        Path to the raw CSV file.
    output_path : str, optional
        Path to save the cleaned dataset. If None, won't save.
    file_format : str, optional
        Format to save the cleaned dataset ('csv', 'parquet', or 'pickle'). Default is 'parquet'.
        
    Returns:
    --------
    pd.DataFrame
        Cleaned and transformed dataset with correct data types.
        
    Returns:
    --------
    pd.DataFrame
        Cleaned and transformed dataset
    """
    print(f"Starting data cleaning process for: {file_path}")
    start_time = datetime.now()
    
    try:
        # Load data
        df_raw = load_data(file_path)
        
        # Convert data types
        df_typed = convert_data_types(df_raw)
        
        # Handle missing values
        df_no_missing = handle_missing_values(df_typed)
        
        # Clean text fields
        df_text_clean = clean_text_fields(df_no_missing)
        
        # Create business metrics
        df_with_metrics = create_business_metrics(df_text_clean)
        
        # Validate data
        validation_results, validation_failed = validate_data(df_with_metrics)
        
        # Generate summary statistics
        summary_stats = create_summary_stats(df_with_metrics)
        
        # Save cleaned data if output path is provided
        if output_path:
            if file_format == 'csv':
                df_with_metrics.to_csv(output_path, index=False)
                print(f"\nCleaned dataset saved as CSV to: {output_path}")
            elif file_format == 'parquet':
                df_with_metrics.to_parquet(output_path, index=False)
                print(f"\nCleaned dataset saved as Parquet to: {output_path}")
            elif file_format == 'pickle':
                df_with_metrics.to_pickle(output_path)
                print(f"\nCleaned dataset saved as Pickle to: {output_path}")
            else:
                raise ValueError("Invalid file_format. Choose 'csv', 'parquet', or 'pickle'.")
            
        # Report execution time
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        print(f"\nData cleaning completed in {execution_time:.2f} seconds")
        
        return df_with_metrics
        
    except Exception as e:
        print(f"ERROR: Data cleaning failed: {e}")
        raise

# Example usage
if __name__ == "__main__":
    cleaned_df = clean_ecommerce_data('Ecommerce_Dataset.csv', 'ecommerce_cleaned.parquet', file_format='parquet')
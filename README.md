# Bank Customer Segmentation Dashboard

A professional Streamlit dashboard for analyzing bank customer transactions and segmentation with a modern matt black, blue, and purple theme.

## 🚀 Key Features

- **Professional Design**: Matt black background with blue and purple accents
- **Date Range Filtering**: Analyze specific time periods with easy date selection
- **Multiple Visualization Types**:
  - Time series charts for transaction volume and amount
  - Geographic distribution of transactions across locations
  - Customer demographic analysis (age, gender distribution)
  - Financial metrics analysis (account balances, transaction amounts)
  - Hourly transaction patterns
  - Customer segmentation scatter plots
- **Responsive Layout**: All charts update automatically when filters change
- **Real-time KPI Metrics**: Key business metrics calculated on-the-fly
- **Data Preview**: Summary statistics and raw data preview

## 📊 Data Columns Utilized

- `TransactionID`: Unique transaction identifier
- `CustomerID`: Unique customer identifier
- `CustomerDOB`: Customer date of birth (used for age calculation)
- `CustGender`: Customer gender (M/F analysis)
- `CustLocation`: Customer location (geographic analysis)
- `CustAccountBalance`: Account balance (financial metrics)
- `TransactionDate`: Transaction date (time series analysis)
- `TransactionTime`: Transaction timestamp (hourly patterns)
- `TransactionAmount (INR)`: Transaction amount in INR (financial analysis)

## 🛠️ Installation

1. Ensure your data file is at: `C:/Users/paras/Desktop/Bank_Customer_Segmentation_Dashboard/data/bank_transactions.csv`

2. Install required packages:

```bash
pip install -r requirements.txt
```

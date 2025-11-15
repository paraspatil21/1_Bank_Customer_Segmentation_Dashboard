import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
import numpy as np

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from data_loader import DataLoader
from visualizations import DashboardVisualizations

class BankCustomerSegmentationDashboard:
    def __init__(self):
        self.set_page_config()
        self.data_loader = None
        self.visualizations = None
        
    def set_page_config(self):
        """Configure the Streamlit page"""
        st.set_page_config(
            page_title="Bank Customer Analytics Dashboard",
            page_icon="",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for professional theme
        st.markdown("""
        <style>
        .main {
            background-color: #0e1a2b;
        }
        .stApp {
            background: linear-gradient(135deg, #0e1a2b 0%, #1a1a2e 50%, #16213e 100%);
        }
        .css-1d391kg, .css-12oz5g7 {
            background: transparent;
        }
        .metric-card {
            background: rgba(30, 40, 60, 0.8);
            padding: 20px;
            border-radius: 12px;
            border-left: 4px solid #1f77b4;
            border-right: 1px solid rgba(255, 255, 255, 0.1);
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            margin: 5px;
            transition: transform 0.2s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            border-left: 4px solid #9467bd;
        }
        .section-header {
            font-size: 28px;
            font-weight: 700;
            color: #1f77b4;
            margin: 40px 0 25px 0;
            padding-bottom: 15px;
            border-bottom: 3px solid #9467bd;
            text-align: center;
        }
        .subsection-header {
            font-size: 22px;
            font-weight: 600;
            color: #9467bd;
            margin: 30px 0 15px 0;
            padding-left: 10px;
            border-left: 4px solid #2ca02c;
        }
        .sidebar .sidebar-content {
            background: rgba(14, 26, 43, 0.95);
            backdrop-filter: blur(10px);
        }
        h1 {
            color: #ffffff;
            border-bottom: 3px solid #9467bd;
            padding-bottom: 20px;
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        h2, h3 {
            color: #ffffff;
        }
        .stButton>button {
            background: linear-gradient(45deg, #1f77b4, #9467bd);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: 600;
        }
        .stDownloadButton>button {
            background: linear-gradient(45deg, #2ca02c, #9467bd);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: 600;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _create_sample_data(self):
        """Create sample data for demonstration when real data is not available"""
        st.warning("Using sample data for demonstration purposes")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate sample data
        n_records = 5000
        customer_ids = [f"CUST{str(i).zfill(5)}" for i in range(1, 501)]
        locations = ['MUMBAI', 'DELHI', 'BANGALORE', 'HYDERABAD', 'CHENNAI', 'KOLKATA', 'PUNE', 'AHMEDABAD']
        
        sample_data = {
            'TransactionID': [f"TXN{str(i).zfill(6)}" for i in range(1, n_records + 1)],
            'CustomerID': np.random.choice(customer_ids, n_records),
            'CustomerDOB': pd.to_datetime(np.random.choice(pd.date_range('1950-01-01', '2000-12-31'), n_records)),
            'CustGender': np.random.choice(['M', 'F'], n_records, p=[0.6, 0.4]),
            'CustLocation': np.random.choice(locations, n_records),
            'CustAccountBalance': np.random.uniform(1000, 500000, n_records),
            'TransactionDate': pd.to_datetime(np.random.choice(pd.date_range('2023-01-01', '2024-01-01'), n_records)),
            'TransactionAmount (INR)': np.random.exponential(5000, n_records)
        }
        
        df = pd.DataFrame(sample_data)
        
        # Calculate derived fields
        df['CustomerAge'] = (df['TransactionDate'].dt.year - df['CustomerDOB'].dt.year)
        df['TransactionHour'] = np.random.randint(0, 24, n_records)
        df['TransactionDayOfWeek'] = df['TransactionDate'].dt.dayofweek
        df['TimeOfDay'] = pd.cut(df['TransactionHour'], 
                                bins=[0, 6, 12, 18, 24], 
                                labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                include_lowest=True)
        
        # Create transaction categories
        conditions = [
            df['TransactionAmount (INR)'] <= 1000,
            (df['TransactionAmount (INR)'] > 1000) & (df['TransactionAmount (INR)'] <= 5000),
            (df['TransactionAmount (INR)'] > 5000) & (df['TransactionAmount (INR)'] <= 10000),
            df['TransactionAmount (INR)'] > 10000
        ]
        choices = ['Small (<1K)', 'Medium (1K-5K)', 'Large (5K-10K)', 'Very Large (>10K)']
        df['TransactionCategory'] = np.select(conditions, choices, default='Unknown')
        
        return df
    
    def load_data(self):
        """Load the data with correct path or create sample data"""
        import os
        # Use relative path for Streamlit Cloud
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, 'data', 'bank_transactions.csv')
        
        # Check if file exists
        if not os.path.exists(data_path):
            st.error(f"Data file not found at: {data_path}")
            st.info("Using sample data for demonstration. To use your own data, please upload 'bank_transactions.csv' to the 'data' folder in your GitHub repository.")
            return self._create_sample_data()
        
        st.info(f"Loading data from: {data_path}")
        try:
            self.data_loader = DataLoader(data_path)
            df = self.data_loader.load_data()
            if df is not None and len(df) > 0:
                st.success(f"Successfully loaded {len(df):,} records")
                return df
            else:
                st.warning("Loaded empty dataset, using sample data instead")
                return self._create_sample_data()
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.info("Using sample data for demonstration")
            return self._create_sample_data()
    
    def create_sidebar_filters(self, df):
        """Create comprehensive sidebar filters"""
        st.sidebar.markdown("## DASHBOARD CONTROLS")
        st.sidebar.markdown("---")
        
        # Date range filter
        st.sidebar.subheader("Date Range")
        min_date = df['TransactionDate'].min().date()
        max_date = df['TransactionDate'].max().date()
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start Date", min_date, key="start_date")
        with col2:
            end_date = st.date_input("End Date", max_date, key="end_date")
        
        st.sidebar.markdown("---")
        
        # Demographic filters
        st.sidebar.subheader("Demographic Filters")
        
        # Age range slider
        min_age = int(df['CustomerAge'].min()) if not df['CustomerAge'].isna().all() else 18
        max_age = int(df['CustomerAge'].max()) if not df['CustomerAge'].isna().all() else 100
        age_range = st.sidebar.slider(
            "Age Range",
            min_value=min_age,
            max_value=max_age,
            value=(min_age, max_age)
        )
        
        # Gender filter
        genders = ['All'] + list(df['CustGender'].unique())
        selected_genders = st.sidebar.multiselect(
            "Gender",
            genders,
            default=['All']
        )
        
        st.sidebar.markdown("---")
        
        # Location filter
        st.sidebar.subheader("Location Filter")
        locations = ['All'] + list(df['CustLocation'].value_counts().head(20).index)
        selected_locations = st.sidebar.multiselect(
            "Select Locations", 
            locations, 
            default=['All']
        )
        
        st.sidebar.markdown("---")
        
        # Financial filters
        st.sidebar.subheader("Financial Filters")
        
        # Balance range
        min_balance = float(df['CustAccountBalance'].min())
        max_balance = float(df['CustAccountBalance'].max())
        balance_range = st.sidebar.slider(
            "Account Balance Range (INR)",
            min_value=min_balance,
            max_value=max_balance,
            value=(min_balance, max_balance)
        )
        
        st.sidebar.markdown("---")
        
        # Analysis settings
        st.sidebar.subheader("Analysis Settings")
        time_frame = st.sidebar.selectbox(
            "Time Frame Analysis",
            ["Daily", "Weekly", "Monthly"],
            index=0
        )
        
        # Performance mode
        performance_mode = st.sidebar.checkbox("Performance Mode (Sample Data)", value=False)
        
        return start_date, end_date, selected_locations, selected_genders, age_range, balance_range, time_frame, performance_mode
    
    def display_kpi_metrics(self, filtered_df):
        """Display comprehensive KPI metrics"""
        st.markdown('<div class="section-header">KEY PERFORMANCE INDICATORS</div>', unsafe_allow_html=True)
        
        # Calculate metrics
        total_transactions = len(filtered_df)
        unique_customers = filtered_df['CustomerID'].nunique()
        total_amount = filtered_df['TransactionAmount (INR)'].sum()
        avg_transaction = filtered_df['TransactionAmount (INR)'].mean()
        avg_balance = filtered_df['CustAccountBalance'].mean()
        max_transaction = filtered_df['TransactionAmount (INR)'].max()
        min_balance = filtered_df['CustAccountBalance'].min()
        max_balance = filtered_df['CustAccountBalance'].max()
        
        # Create metrics in two rows
        col1, col2, col3, col4 = st.columns(4)
        col5, col6, col7, col8 = st.columns(4)
        
        metrics_row1 = [
            {'title': 'Total Transactions', 'value': f"{total_transactions:,}", 'color': '#1f77b4'},
            {'title': 'Unique Customers', 'value': f"{unique_customers:,}", 'color': '#9467bd'},
            {'title': 'Total Amount', 'value': f"INR{total_amount:,.0f}", 'color': '#2ca02c'},
            {'title': 'Avg Transaction', 'value': f"INR{avg_transaction:,.0f}", 'color': '#ff7f0e'},
        ]
        
        metrics_row2 = [
            {'title': 'Avg Balance', 'value': f"INR{avg_balance:,.0f}", 'color': '#d62728'},
            {'title': 'Max Transaction', 'value': f"INR{max_transaction:,.0f}", 'color': '#8c564b'},
            {'title': 'Min Balance', 'value': f"INR{min_balance:,.0f}", 'color': '#e377c2'},
            {'title': 'Max Balance', 'value': f"INR{max_balance:,.0f}", 'color': '#7f7f7f'},
        ]
        
        for i, metric in enumerate(metrics_row1):
            with [col1, col2, col3, col4][i]:
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                        <h3 style='color: {metric['color']}; margin:0; font-size: 14px;'>{metric['title']}</h3>
                    </div>
                    <h2 style='color: white; margin:0; font-size: 20px; font-weight: bold;'>{metric['value']}</h2>
                </div>
                """, unsafe_allow_html=True)
        
        for i, metric in enumerate(metrics_row2):
            with [col5, col6, col7, col8][i]:
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                        <h3 style='color: {metric['color']}; margin:0; font-size: 14px;'>{metric['title']}</h3>
                    </div>
                    <h2 style='color: white; margin:0; font-size: 20px; font-weight: bold;'>{metric['value']}</h2>
                </div>
                """, unsafe_allow_html=True)
    
    def display_geospatial_analysis(self, filtered_df):
        """Display comprehensive geospatial analysis"""
        st.markdown('<div class="section-header">GEOSPATIAL ANALYSIS</div>', unsafe_allow_html=True)
        
        # Comprehensive Geospatial Dashboard
        st.markdown('<div class="subsection-header">India Transaction Map & Regional Analysis</div>', unsafe_allow_html=True)
        fig1 = self.visualizations.create_geospatial_visualizations()
        st.plotly_chart(fig1, use_container_width=True, key="geo_1")
        
        # Interactive India Map
        st.markdown('<div class="subsection-header">Interactive India Transaction Map</div>', unsafe_allow_html=True)
        fig2 = self.visualizations.create_interactive_india_map()
        st.plotly_chart(fig2, use_container_width=True, key="geo_2")
        
        # Regional Analysis
        st.markdown('<div class="subsection-header">Regional Performance Analysis</div>', unsafe_allow_html=True)
        fig3 = self.visualizations.create_regional_analysis()
        st.plotly_chart(fig3, use_container_width=True, key="geo_3")
    
    def display_transaction_analysis(self, filtered_df, time_frame):
        """Display comprehensive transaction analysis"""
        st.markdown('<div class="section-header">TRANSACTION ANALYSIS</div>', unsafe_allow_html=True)
        
        # Transaction volume and trends
        fig1 = self.visualizations.create_transaction_volume_chart(time_frame)
        st.plotly_chart(fig1, use_container_width=True, key="txn_1")
        
        # Geographic analysis
        st.markdown('<div class="subsection-header">Geographic Distribution</div>', unsafe_allow_html=True)
        fig2 = self.visualizations.create_geographic_heatmap()
        st.plotly_chart(fig2, use_container_width=True, key="txn_2")
    
    def display_customer_analysis(self, filtered_df):
        """Display comprehensive customer analysis"""
        st.markdown('<div class="section-header">CUSTOMER ANALYSIS</div>', unsafe_allow_html=True)
        
        # Customer demographics
        st.markdown('<div class="subsection-header">Customer Demographics</div>', unsafe_allow_html=True)
        fig1 = self.visualizations.create_customer_demographics()
        st.plotly_chart(fig1, use_container_width=True, key="cust_1")
        
        # Behavioral analysis
        st.markdown('<div class="subsection-header">Behavioral Patterns</div>', unsafe_allow_html=True)
        fig2 = self.visualizations.create_behavioral_analysis()
        st.plotly_chart(fig2, use_container_width=True, key="cust_2")
    
    def display_financial_analysis(self, filtered_df):
        """Display comprehensive financial analysis"""
        st.markdown('<div class="section-header">FINANCIAL ANALYSIS</div>', unsafe_allow_html=True)
        
        fig = self.visualizations.create_financial_metrics()
        st.plotly_chart(fig, use_container_width=True, key="finance_1")
    
    def display_advanced_segmentation(self, filtered_df):
        """Display advanced customer segmentation"""
        st.markdown('<div class="section-header">ADVANCED CUSTOMER SEGMENTATION</div>', unsafe_allow_html=True)
        
        # Simple segmentation overview
        st.markdown('<div class="subsection-header">Segmentation Overview</div>', unsafe_allow_html=True)
        fig1 = self.visualizations.create_simple_segmentation()
        st.plotly_chart(fig1, use_container_width=True, key="seg_1")
        
        # RFM Analysis
        st.markdown('<div class="subsection-header">RFM Analysis (Recency, Frequency, Monetary)</div>', unsafe_allow_html=True)
        fig2 = self.visualizations.create_rfm_analysis()
        st.plotly_chart(fig2, use_container_width=True, key="seg_2")
        
        # Advanced clustering
        st.markdown('<div class="subsection-header">K-means Clustering (3D Visualization)</div>', unsafe_allow_html=True)
        fig3 = self.visualizations.create_advanced_segmentation()
        st.plotly_chart(fig3, use_container_width=True, key="seg_3")
    
    def display_data_insights(self, filtered_df):
        """Display data insights and summary"""
        st.markdown('<div class="section-header">DATA INSIGHTS & SUMMARY</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="subsection-header">Key Statistics</div>', unsafe_allow_html=True)
            
            insights_data = {
                'Metric': [
                    'Total Transaction Volume',
                    'Average Daily Transactions',
                    'Customer Acquisition Rate',
                    'Peak Transaction Hour',
                    'Most Active Location',
                    'Gender Distribution Ratio',
                    'Average Customer Age',
                    'Balance to Transaction Ratio'
                ],
                'Value': [
                    f"INR{filtered_df['TransactionAmount (INR)'].sum():,.0f}",
                    f"{len(filtered_df) // max(1, (filtered_df['TransactionDate'].max() - filtered_df['TransactionDate'].min()).days):,}",
                    f"{filtered_df['CustomerID'].nunique() // max(1, (filtered_df['TransactionDate'].max() - filtered_df['TransactionDate'].min()).days):,}",
                    f"{filtered_df['TransactionHour'].mode().iloc[0] if not filtered_df['TransactionHour'].mode().empty else 'N/A'} Hrs",
                    f"{filtered_df['CustLocation'].mode().iloc[0] if not filtered_df['CustLocation'].mode().empty else 'N/A'}",
                    f"M:{(filtered_df['CustGender'] == 'M').sum():,} | F:{(filtered_df['CustGender'] == 'F').sum():,}",
                    f"{filtered_df['CustomerAge'].mean():.1f} years",
                    f"{(filtered_df['CustAccountBalance'].mean() / max(1, filtered_df['TransactionAmount (INR)'].mean())):.2f}"
                ]
            }
            
            insights_df = pd.DataFrame(insights_data)
            st.dataframe(insights_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown('<div class="subsection-header">Data Preview</div>', unsafe_allow_html=True)
            
            # Sample data with more columns
            display_columns = [
                'TransactionID', 'CustomerID', 'CustGender', 'CustLocation',
                'TransactionAmount (INR)', 'CustAccountBalance', 'TransactionDate', 'TransactionHour'
            ]
            
            sample_data = filtered_df[display_columns].head(10)
            st.dataframe(sample_data, use_container_width=True)
            
            # Data quality indicators
            st.markdown('<div class="subsection-header">Data Quality</div>', unsafe_allow_html=True)
            
            quality_metrics = {
                'Metric': ['Total Records', 'Complete Records', 'Missing Values', 'Duplicate Transactions'],
                'Count': [
                    len(filtered_df),
                    filtered_df.notna().all(axis=1).sum(),
                    filtered_df.isna().sum().sum(),
                    filtered_df.duplicated().sum()
                ]
            }
            
            quality_df = pd.DataFrame(quality_metrics)
            st.dataframe(quality_df, use_container_width=True, hide_index=True)
    
    def run(self):
        """Run the enhanced dashboard"""
        # Enhanced Header
        st.markdown("""
        <h1 style='text-align: center; margin-bottom: 10px;'>
            Advanced Bank Customer Analytics Dashboard
        </h1>
        <p style='text-align: center; color: #cccccc; font-size: 18px; margin-bottom: 40px;'>
            Comprehensive Customer Segmentation & Geospatial Analysis Platform
        </p>
        """, unsafe_allow_html=True)
        
        # Load data with progress
        with st.spinner('Loading and processing data... This may take a moment for advanced analytics.'):
            df = self.load_data()
        
        if df is None or len(df) == 0:
            st.error("Failed to load data. The dashboard cannot function without data.")
            return
        
        # Initialize data loader with the loaded dataframe
        self.data_loader = DataLoader(None)
        self.data_loader.df = df
        self.data_loader._is_loaded = True
        
        # Create enhanced filters
        start_date, end_date, locations, genders, age_range, balance_range, time_frame, performance_mode = self.create_sidebar_filters(df)
        
        # Filter data
        filtered_df = self.data_loader.get_filtered_data(
            start_date=pd.to_datetime(start_date),
            end_date=pd.to_datetime(end_date),
            locations=locations,
            genders=genders,
            age_range=age_range,
            balance_range=balance_range
        )
        
        # Apply performance mode sampling
        if performance_mode and len(filtered_df) > 10000:
            filtered_df = filtered_df.sample(n=10000, random_state=42)
            st.info(f"Performance Mode: Using 10,000 sampled records from {len(filtered_df):,} total records")
        
        # Store filtered data for export
        self.current_filtered_df = filtered_df
        
        # Get customer segments and geospatial data
        customer_segments = self.data_loader.get_customer_segments()
        geo_data = self.data_loader.get_geospatial_data()
        
        # Initialize enhanced visualizations with geospatial data
        self.visualizations = DashboardVisualizations(filtered_df, customer_segments, geo_data)
        
        # Display all dashboard sections
        self.display_kpi_metrics(filtered_df)
        self.display_geospatial_analysis(filtered_df)
        self.display_transaction_analysis(filtered_df, time_frame)
        self.display_customer_analysis(filtered_df)
        self.display_financial_analysis(filtered_df)
        self.display_advanced_segmentation(filtered_df)
        self.display_data_insights(filtered_df)
        
        # Enhanced Footer
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col2:
            st.markdown("""
            <div style='text-align: center; color: #888;'>
                <p style='margin: 5px 0;'>Advanced Bank Customer Analytics Dashboard</p>
                <p style='margin: 5px 0;'>Built with Streamlit & Plotly</p>
                <p style='margin: 5px 0;'>Real-time Analytics • Geospatial Insights • Customer Segmentation</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    dashboard = BankCustomerSegmentationDashboard()
    dashboard.run()
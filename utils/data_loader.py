import pandas as pd
import numpy as np
from datetime import datetime
import os
import streamlit as st
import geopandas as gpd
from shapely.geometry import Point
import requests
import json

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.customer_segments = None
        self._is_loaded = False
        self.geo_data = None
        
    def load_data(self):
        """Load and preprocess the bank transactions data"""
        try:
            if self._is_loaded and self.df is not None:
                return self.df
                
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("📂 Loading CSV file...")
            # Load with optimized parameters
            self.df = pd.read_csv(
                self.file_path,
                dtype={
                    'TransactionID': 'string',
                    'CustomerID': 'string',
                    'CustGender': 'category',
                    'CustLocation': 'category'
                },
                parse_dates=['TransactionDate'],
                infer_datetime_format=True,
                low_memory=False
            )
            progress_bar.progress(40)
            
            status_text.text("🔄 Preprocessing data...")
            self._preprocess_data()
            progress_bar.progress(60)
            
            status_text.text("🗺️ Processing geospatial data...")
            self._process_geospatial_data()
            progress_bar.progress(80)
            
            status_text.text("📊 Creating customer segments...")
            self._create_customer_segments()
            progress_bar.progress(90)
            
            status_text.text("✅ Data loaded successfully!")
            progress_bar.progress(100)
            status_text.empty()
            
            self._is_loaded = True
            return self.df
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    
    def _preprocess_data(self):
        """Preprocess the data for analysis"""
        # Convert dates
        self.df['CustomerDOB'] = pd.to_datetime(self.df['CustomerDOB'], errors='coerce')
        
        # Calculate customer age
        self.df['CustomerAge'] = (self.df['TransactionDate'].dt.year - self.df['CustomerDOB'].dt.year)
        self.df['CustomerAge'] = self.df['CustomerAge'].apply(
            lambda x: x if pd.notna(x) and 18 <= x <= 100 else np.nan
        )
        
        # Extract additional features
        if 'TransactionTime' in self.df.columns:
            self.df['TransactionHour'] = pd.to_datetime(
                self.df['TransactionTime'], unit='s', errors='coerce'
            ).dt.hour.fillna(self.df['TransactionDate'].dt.hour)
        else:
            self.df['TransactionHour'] = self.df['TransactionDate'].dt.hour
            
        self.df['TransactionDay'] = self.df['TransactionDate'].dt.day_name()
        self.df['TransactionMonth'] = self.df['TransactionDate'].dt.month_name()
        self.df['TransactionYear'] = self.df['TransactionDate'].dt.year
        self.df['TransactionDayOfWeek'] = self.df['TransactionDate'].dt.dayofweek
        
        # Create time-based segments
        try:
            self.df['TimeOfDay'] = pd.cut(self.df['TransactionHour'], 
                                         bins=[0, 6, 12, 18, 24], 
                                         labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                         include_lowest=True,
                                         duplicates='drop')
        except ValueError:
            # Fallback if there are still duplicate issues
            self.df['TimeOfDay'] = pd.cut(self.df['TransactionHour'], 
                                         bins=4, 
                                         labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                         include_lowest=True)
        
        # Create transaction categories
        conditions = [
            self.df['TransactionAmount (INR)'] <= 1000,
            (self.df['TransactionAmount (INR)'] > 1000) & (self.df['TransactionAmount (INR)'] <= 5000),
            (self.df['TransactionAmount (INR)'] > 5000) & (self.df['TransactionAmount (INR)'] <= 10000),
            self.df['TransactionAmount (INR)'] > 10000
        ]
        choices = ['Small (<1K)', 'Medium (1K-5K)', 'Large (5K-10K)', 'Very Large (>10K)']
        self.df['TransactionCategory'] = np.select(conditions, choices, default='Unknown')
        
        # Create balance categories
        balance_conditions = [
            self.df['CustAccountBalance'] <= 10000,
            (self.df['CustAccountBalance'] > 10000) & (self.df['CustAccountBalance'] <= 50000),
            (self.df['CustAccountBalance'] > 50000) & (self.df['CustAccountBalance'] <= 100000),
            (self.df['CustAccountBalance'] > 100000) & (self.df['CustAccountBalance'] <= 500000),
            self.df['CustAccountBalance'] > 500000
        ]
        balance_choices = ['Low (<10K)', 'Medium (10K-50K)', 'High (50K-100K)', 'Premium (100K-500K)', 'Elite (>500K)']
        self.df['BalanceCategory'] = np.select(balance_conditions, balance_choices, default='Unknown')
        
        # Clean numeric columns
        numeric_columns = ['CustAccountBalance', 'TransactionAmount (INR)']
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(0)
    
    def _process_geospatial_data(self):
        """Process geospatial data for Indian cities"""
        try:
            # Create a mapping of Indian cities to approximate coordinates
            indian_cities_coords = {
                'MUMBAI': (19.0760, 72.8777),
                'DELHI': (28.7041, 77.1025),
                'BANGALORE': (12.9716, 77.5946),
                'HYDERABAD': (17.3850, 78.4867),
                'AHMEDABAD': (23.0225, 72.5714),
                'CHENNAI': (13.0827, 80.2707),
                'KOLKATA': (22.5726, 88.3639),
                'SURAT': (21.1702, 72.8311),
                'PUNE': (18.5204, 73.8567),
                'JAIPUR': (26.9124, 75.7873),
                'LUCKNOW': (26.8467, 80.9462),
                'KANPUR': (26.4499, 80.3319),
                'NAGPUR': (21.1458, 79.0882),
                'INDORE': (22.7196, 75.8577),
                'THANE': (19.2183, 72.9781),
                'BHOPAL': (23.2599, 77.4126),
                'VISAKHAPATNAM': (17.6868, 83.2185),
                'PATNA': (25.5941, 85.1376),
                'VADODARA': (22.3072, 73.1812),
                'GHAZIABAD': (28.6692, 77.4538),
                'LUDHIANA': (30.9010, 75.8573),
                'AGRA': (27.1767, 78.0081),
                'NASHIK': (19.9975, 73.7898),
                'FARIDABAD': (28.4089, 77.3178),
                'MEERUT': (28.9845, 77.7064),
                'RAJKOT': (22.3039, 70.8022),
                'KALYAN': (19.2437, 73.1355),
                'VASANT KUNJ': (28.4595, 77.0737),
                'VARANASI': (25.3176, 82.9739),
                'SRINAGAR': (34.0837, 74.7973)
            }
            
            # Add coordinates to the dataframe
            self.df['City'] = self.df['CustLocation'].str.upper().str.strip()
            self.df['Latitude'] = self.df['City'].map({city: coords[0] for city, coords in indian_cities_coords.items()})
            self.df['Longitude'] = self.df['City'].map({city: coords[1] for city, coords in indian_cities_coords.items()})
            
            # Create geometry for geopandas
            self.df['geometry'] = self.df.apply(
                lambda row: Point(row['Longitude'], row['Latitude']) 
                if pd.notna(row['Longitude']) and pd.notna(row['Latitude']) 
                else None, 
                axis=1
            )
            
            # Create geodataframe for spatial operations
            valid_geo_data = self.df.dropna(subset=['geometry'])
            if len(valid_geo_data) > 0:
                self.geo_data = gpd.GeoDataFrame(valid_geo_data, geometry='geometry')
                self.geo_data.crs = "EPSG:4326"
            
        except Exception as e:
            st.warning(f"Geospatial processing limited: {e}")
            self.geo_data = None
    
    def _create_customer_segments(self):
        """Create customer segments for RFM analysis"""
        try:
            # Sample data if too large for faster processing
            if len(self.df) > 100000:
                customer_stats = self.df.groupby('CustomerID').agg({
                    'TransactionID': 'count',
                    'TransactionAmount (INR)': ['sum', 'mean'],
                    'CustAccountBalance': 'last',
                    'TransactionDate': ['min', 'max']
                }).reset_index()
            else:
                customer_stats = self.df.groupby('CustomerID').agg({
                    'TransactionID': 'count',
                    'TransactionAmount (INR)': ['sum', 'mean'],
                    'CustAccountBalance': 'last',
                    'TransactionDate': ['min', 'max']
                }).reset_index()
            
            # Flatten column names
            customer_stats.columns = ['CustomerID', 'TransactionCount', 'TotalAmount', 'AvgTransaction', 
                                     'LastBalance', 'FirstTransaction', 'LastTransaction']
            
            # Calculate recency
            max_date = self.df['TransactionDate'].max()
            customer_stats['Recency'] = (max_date - customer_stats['LastTransaction']).dt.days
            customer_stats['CustomerTenure'] = (customer_stats['LastTransaction'] - customer_stats['FirstTransaction']).dt.days
            
            # Create RFM segments with error handling
            try:
                customer_stats['FrequencyScore'] = pd.qcut(customer_stats['TransactionCount'], 
                                                          q=3, 
                                                          labels=['Low', 'Medium', 'High'],
                                                          duplicates='drop')
                customer_stats['MonetaryScore'] = pd.qcut(customer_stats['TotalAmount'], 
                                                         q=3, 
                                                         labels=['Low', 'Medium', 'High'],
                                                         duplicates='drop')
                customer_stats['RecencyScore'] = pd.qcut(customer_stats['Recency'], 
                                                        q=3, 
                                                        labels=['High', 'Medium', 'Low'],
                                                        duplicates='drop')
            except ValueError:
                # Fallback manual segmentation
                customer_stats['FrequencyScore'] = np.where(
                    customer_stats['TransactionCount'] > customer_stats['TransactionCount'].median(), 'High', 'Low'
                )
                customer_stats['MonetaryScore'] = np.where(
                    customer_stats['TotalAmount'] > customer_stats['TotalAmount'].median(), 'High', 'Low'
                )
                customer_stats['RecencyScore'] = np.where(
                    customer_stats['Recency'] < customer_stats['Recency'].median(), 'High', 'Low'
                )
            
            self.customer_segments = customer_stats
            
        except Exception as e:
            st.warning(f"Could not create customer segments: {e}")
            self.customer_segments = None
    
    def get_filtered_data(self, start_date=None, end_date=None, locations=None, genders=None, age_range=None, balance_range=None):
        """Filter data based on user selections"""
        if self.df is None:
            st.error("Data not loaded. Please load data first.")
            return pd.DataFrame()
            
        filtered_df = self.df.copy()
        
        if start_date:
            filtered_df = filtered_df[filtered_df['TransactionDate'] >= pd.to_datetime(start_date)]
        if end_date:
            filtered_df = filtered_df[filtered_df['TransactionDate'] <= pd.to_datetime(end_date)]
        if locations and 'All' not in locations:
            filtered_df = filtered_df[filtered_df['CustLocation'].isin(locations)]
        if genders and 'All' not in genders:
            filtered_df = filtered_df[filtered_df['CustGender'].isin(genders)]
        if age_range:
            filtered_df = filtered_df[
                (filtered_df['CustomerAge'] >= age_range[0]) & 
                (filtered_df['CustomerAge'] <= age_range[1])
            ]
        if balance_range:
            filtered_df = filtered_df[
                (filtered_df['CustAccountBalance'] >= balance_range[0]) & 
                (filtered_df['CustAccountBalance'] <= balance_range[1])
            ]
            
        return filtered_df
    
    def get_customer_segments(self):
        """Get customer segmentation data"""
        return self.customer_segments
    
    def get_geospatial_data(self):
        """Get geospatial data"""
        return self.geo_data
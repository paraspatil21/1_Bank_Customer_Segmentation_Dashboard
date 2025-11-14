import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class DashboardVisualizations:
    def __init__(self, data, customer_segments=None, geo_data=None):
        self.data = data
        self.customer_segments = customer_segments
        self.geo_data = geo_data
        self.colors = {
            'primary': '#1f77b4',  # Blue
            'secondary': '#9467bd', # Purple
            'accent': '#2ca02c',   # Green
            'warning': '#ff7f0e',  # Orange
            'danger': '#d62728',   # Red
            'background': '#0e1a2b',
            'text': '#ffffff'
        }
    
    def create_geospatial_visualizations(self):
        """Create comprehensive geospatial visualizations"""
        try:
            if self.geo_data is None or len(self.geo_data) == 0:
                return self._create_empty_plot("No geospatial data available")
            
            # Sample for performance
            if len(self.geo_data) > 10000:
                plot_data = self.geo_data.sample(n=5000, random_state=42)
            else:
                plot_data = self.geo_data
            
            # Create subplots for geospatial analysis
            fig = sp.make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Transaction Density Map - India',
                    'Transaction Amount by City',
                    'Customer Distribution Heatmap', 
                    'Top Cities by Transaction Volume'
                ),
                specs=[
                    [{"type": "scattermapbox"}, {"type": "bar"}],
                    [{"type": "densitymapbox"}, {"type": "bar"}]
                ],
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            # 1. Scatter Map - Transaction Locations
            city_stats = plot_data.groupby('City').agg({
                'TransactionID': 'count',
                'TransactionAmount (INR)': 'mean',
                'Latitude': 'first',
                'Longitude': 'first'
            }).reset_index()
            
            fig.add_trace(
                go.Scattermapbox(
                    lat=city_stats['Latitude'],
                    lon=city_stats['Longitude'],
                    mode='markers',
                    marker=dict(
                        size=city_stats['TransactionID'] / city_stats['TransactionID'].max() * 50 + 10,
                        color=city_stats['TransactionAmount (INR)'],
                        colorscale='Viridis',
                        colorbar=dict(title="Avg Amount"),
                        opacity=0.7
                    ),
                    text=city_stats['City'] + '<br>Transactions: ' + city_stats['TransactionID'].astype(str) + 
                         '<br>Avg Amount: ₹' + city_stats['TransactionAmount (INR)'].round(2).astype(str),
                    hoverinfo='text',
                    name='Transaction Centers'
                ),
                row=1, col=1
            )
            
            # 2. Bar chart - Top cities by transaction count
            top_cities = city_stats.nlargest(10, 'TransactionID')
            fig.add_trace(
                go.Bar(
                    x=top_cities['TransactionID'],
                    y=top_cities['City'],
                    orientation='h',
                    marker_color=self.colors['primary'],
                    name='Transaction Count'
                ),
                row=1, col=2
            )
            
            # 3. Density Map - Customer distribution
            fig.add_trace(
                go.Densitymapbox(
                    lat=plot_data['Latitude'],
                    lon=plot_data['Longitude'],
                    z=plot_data['TransactionAmount (INR)'],
                    radius=20,
                    colorscale='Hot',
                    colorbar=dict(title="Amount Density"),
                    name='Transaction Density'
                ),
                row=2, col=1
            )
            
            # 4. Bar chart - Top cities by average transaction amount
            top_amount_cities = city_stats.nlargest(10, 'TransactionAmount (INR)')
            fig.add_trace(
                go.Bar(
                    x=top_amount_cities['TransactionAmount (INR)'],
                    y=top_amount_cities['City'],
                    orientation='h',
                    marker_color=self.colors['secondary'],
                    name='Avg Amount'
                ),
                row=2, col=2
            )
            
            # Update map layout
            fig.update_layout(
                mapbox1=dict(
                    style="carto-positron",
                    center=dict(lat=20.5937, lon=78.9629),  # Center of India
                    zoom=3.5
                ),
                mapbox2=dict(
                    style="carto-positron",
                    center=dict(lat=20.5937, lon=78.9629),
                    zoom=3.5
                ),
                title=dict(text="Geospatial Analysis - India", x=0.5, font=dict(size=20)),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=self.colors['text']),
                height=800,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_plot(f"Error in geospatial visualization: {str(e)}")
    
    def create_interactive_india_map(self):
        """Create an interactive India map with transaction data"""
        try:
            if self.geo_data is None or len(self.geo_data) == 0:
                return self._create_empty_plot("No geospatial data available")
            
            # Aggregate data by city
            city_data = self.geo_data.groupby('City').agg({
                'TransactionID': 'count',
                'TransactionAmount (INR)': ['sum', 'mean', 'max'],
                'CustomerID': 'nunique',
                'Latitude': 'first',
                'Longitude': 'first'
            }).reset_index()
            
            city_data.columns = ['City', 'TransactionCount', 'TotalAmount', 'AvgAmount', 'MaxAmount', 'UniqueCustomers', 'Latitude', 'Longitude']
            
            # Create bubble map
            fig = px.scatter_mapbox(
                city_data,
                lat="Latitude",
                lon="Longitude",
                size="TransactionCount",
                color="AvgAmount",
                hover_name="City",
                hover_data={
                    'TransactionCount': True,
                    'TotalAmount': ':.2f',
                    'AvgAmount': ':.2f',
                    'UniqueCustomers': True,
                    'Latitude': False,
                    'Longitude': False
                },
                color_continuous_scale="Viridis",
                size_max=50,
                zoom=3.5,
                height=600,
                title="Interactive India Transaction Map"
            )
            
            fig.update_layout(
                mapbox_style="carto-positron",
                mapbox=dict(
                    center=dict(lat=20.5937, lon=78.9629)
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=self.colors['text'])
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_plot(f"Error in interactive map: {str(e)}")
    
    def create_regional_analysis(self):
        """Create regional analysis charts"""
        try:
            if self.geo_data is None:
                return self._create_empty_plot("No geospatial data available")
            
            # Define Indian regions
            region_mapping = {
                'NORTH': ['DELHI', 'GURGAON', 'NOIDA', 'FARIDABAD', 'GHAZIABAD', 'LUCKNOW', 'KANPUR', 'JAIPUR', 'LUDHIANA', 'AMRITSAR'],
                'SOUTH': ['BANGALORE', 'HYDERABAD', 'CHENNAI', 'COIMBATORE', 'KOCHI', 'VIZAG', 'VIJAYAWADA', 'MYSORE', 'MADURAI'],
                'WEST': ['MUMBAI', 'PUNE', 'AHMEDABAD', 'SURAT', 'VADODARA', 'NAGPUR', 'INDORE', 'BHOPAL', 'GOA'],
                'EAST': ['KOLKATA', 'PATNA', 'BHUBANESWAR', 'RANCHI', 'GUWAHATI', 'SHILLONG', 'IMPHAL']
            }
            
            # Map cities to regions
            def get_region(city):
                for region, cities in region_mapping.items():
                    if city in cities:
                        return region
                return 'OTHER'
            
            regional_data = self.geo_data.copy()
            regional_data['Region'] = regional_data['City'].apply(get_region)
            
            fig = sp.make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Transaction Volume by Region',
                    'Average Transaction Amount by Region',
                    'Customer Distribution by Region',
                    'Regional Performance Metrics'
                ),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                      [{"type": "pie"}, {"type": "bar"}]]
            )
            
            # 1. Transaction volume by region
            region_volume = regional_data.groupby('Region')['TransactionID'].count().reset_index()
            fig.add_trace(
                go.Bar(x=region_volume['Region'], y=region_volume['TransactionID'],
                      name='Transaction Volume', marker_color=self.colors['primary']),
                row=1, col=1
            )
            
            # 2. Average amount by region
            region_avg_amount = regional_data.groupby('Region')['TransactionAmount (INR)'].mean().reset_index()
            fig.add_trace(
                go.Bar(x=region_avg_amount['Region'], y=region_avg_amount['TransactionAmount (INR)'],
                      name='Avg Amount', marker_color=self.colors['secondary']),
                row=1, col=2
            )
            
            # 3. Customer distribution by region
            region_customers = regional_data.groupby('Region')['CustomerID'].nunique().reset_index()
            fig.add_trace(
                go.Pie(labels=region_customers['Region'], values=region_customers['CustomerID'],
                      name='Customer Distribution', marker_colors=[self.colors['primary'], self.colors['secondary'], 
                                                                 self.colors['accent'], self.colors['warning']]),
                row=2, col=1
            )
            
            # 4. Regional performance (multiple metrics)
            region_performance = regional_data.groupby('Region').agg({
                'TransactionID': 'count',
                'TransactionAmount (INR)': 'sum',
                'CustomerID': 'nunique'
            }).reset_index()
            
            fig.add_trace(
                go.Bar(name='Transactions', x=region_performance['Region'], y=region_performance['TransactionID'],
                      marker_color=self.colors['primary']),
                row=2, col=2
            )
            fig.add_trace(
                go.Bar(name='Total Amount', x=region_performance['Region'], y=region_performance['TransactionAmount (INR)']/1000,
                      marker_color=self.colors['secondary']),
                row=2, col=2
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=self.colors['text']),
                height=700,
                showlegend=True,
                barmode='group'
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_plot(f"Error in regional analysis: {str(e)}")
    
    # ... (Keep all the previous visualization methods exactly as they were)
    # Transaction Volume Chart, Customer Demographics, Behavioral Analysis, 
    # Financial Metrics, Advanced Segmentation, RFM Analysis, etc.
    # ... [All previous methods remain unchanged]
    
    def create_transaction_volume_chart(self, time_frame='Daily'):
        """Create transaction volume over time chart"""
        try:
            # Sample data for large datasets
            if len(self.data) > 50000:
                plot_data = self.data.sample(n=20000, random_state=42)
            else:
                plot_data = self.data
            
            if time_frame == 'Daily':
                df_agg = plot_data.groupby('TransactionDate').agg({
                    'TransactionID': 'count',
                    'TransactionAmount (INR)': 'sum',
                    'CustomerID': 'nunique'
                }).reset_index()
                title = 'Daily Transaction Trends'
            elif time_frame == 'Monthly':
                df_agg = plot_data.groupby(pd.Grouper(key='TransactionDate', freq='M')).agg({
                    'TransactionID': 'count',
                    'TransactionAmount (INR)': 'sum',
                    'CustomerID': 'nunique'
                }).reset_index()
                title = 'Monthly Transaction Trends'
            else:  # Weekly
                df_agg = plot_data.groupby(pd.Grouper(key='TransactionDate', freq='W')).agg({
                    'TransactionID': 'count',
                    'TransactionAmount (INR)': 'sum',
                    'CustomerID': 'nunique'
                }).reset_index()
                title = 'Weekly Transaction Trends'
            
            fig = sp.make_subplots(
                rows=2, cols=1,
                subplot_titles=('Transaction Volume and Amount', 'Unique Customers'),
                vertical_spacing=0.1
            )
            
            # Transaction volume and amount
            fig.add_trace(
                go.Scatter(x=df_agg['TransactionDate'], y=df_agg['TransactionID'], 
                          name="Transaction Count", line=dict(color=self.colors['primary'], width=3)),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=df_agg['TransactionDate'], y=df_agg['TransactionAmount (INR)'], 
                          name="Transaction Amount (INR)", line=dict(color=self.colors['secondary'], width=3),
                          yaxis='y2'),
                row=1, col=1
            )
            
            # Unique customers
            fig.add_trace(
                go.Bar(x=df_agg['TransactionDate'], y=df_agg['CustomerID'],
                      name="Unique Customers", marker_color=self.colors['accent']),
                row=2, col=1
            )
            
            fig.update_layout(
                title=dict(text=title, x=0.5, font=dict(size=16, color=self.colors['text'])),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=self.colors['text']),
                height=600,
                showlegend=True
            )
            
            fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
            fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
            fig.update_yaxes(title_text="Count/Amount", row=1, col=1)
            fig.update_yaxes(title_text="Unique Customers", row=2, col=1)
            
            return fig
        except Exception as e:
            return self._create_empty_plot(f"Error in transaction volume chart: {str(e)}")
    
    def create_geographic_heatmap(self):
        """Create geographic heatmap of transactions"""
        try:
            location_stats = self.data.groupby('CustLocation').agg({
                'TransactionID': 'count',
                'TransactionAmount (INR)': 'sum',
                'CustomerID': 'nunique'
            }).reset_index()
            
            location_stats.columns = ['Location', 'TransactionCount', 'TotalAmount', 'UniqueCustomers']
            
            # Create heatmap data
            heatmap_data = location_stats.nlargest(15, 'TransactionCount')
            
            fig = px.density_heatmap(
                location_stats,
                x='Location',
                y='TransactionCount',
                z='TotalAmount',
                title='Geographic Heatmap - Transaction Density',
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=self.colors['text']),
                height=500,
                xaxis=dict(tickangle=45)
            )
            
            return fig
        except Exception as e:
            return self._create_empty_plot(f"Error in geographic heatmap: {str(e)}")
    
    def create_customer_demographics(self):
        """Create comprehensive customer demographics charts"""
        try:
            fig = sp.make_subplots(
                rows=2, cols=2,
                subplot_titles=('Age Distribution', 'Gender Distribution', 
                              'Age vs Balance', 'Gender vs Transaction Amount'),
                specs=[[{"type": "histogram"}, {"type": "pie"}],
                      [{"type": "scatter"}, {"type": "box"}]]
            )
            
            # Age distribution
            valid_ages = self.data[self.data['CustomerAge'].notna()]
            fig.add_trace(
                go.Histogram(x=valid_ages['CustomerAge'], name='Age Distribution',
                           marker_color=self.colors['primary']),
                row=1, col=1
            )
            
            # Gender distribution - PIE CHART
            gender_counts = self.data['CustGender'].value_counts()
            fig.add_trace(
                go.Pie(labels=gender_counts.index, values=gender_counts.values,
                      name='Gender Distribution', 
                      marker_colors=[self.colors['primary'], self.colors['secondary']]),
                row=1, col=2
            )
            
            # Age vs Balance scatter
            valid_data = self.data[self.data['CustomerAge'].notna() & 
                                 (self.data['CustAccountBalance'] > 0)]
            # Sample for performance
            if len(valid_data) > 10000:
                valid_data = valid_data.sample(n=5000, random_state=42)
                
            fig.add_trace(
                go.Scatter(x=valid_data['CustomerAge'], y=valid_data['CustAccountBalance'],
                          mode='markers', name='Age vs Balance',
                          marker=dict(color=self.colors['accent'], size=5, opacity=0.6)),
                row=2, col=1
            )
            
            # Gender vs Transaction Amount - BOX PLOT
            # Sample for performance
            plot_data = self.data.sample(n=min(5000, len(self.data)), random_state=42) if len(self.data) > 5000 else self.data
            fig.add_trace(
                go.Box(x=plot_data['CustGender'], y=plot_data['TransactionAmount (INR)'],
                      name='Transaction Amount by Gender',
                      marker_color=self.colors['warning']),
                row=2, col=2
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=self.colors['text']),
                height=700,
                showlegend=False
            )
            
            return fig
        except Exception as e:
            return self._create_empty_plot(f"Error in demographics chart: {str(e)}")
    
    def create_advanced_segmentation(self):
        """Create advanced customer segmentation using K-means clustering"""
        try:
            if self.customer_segments is None or len(self.customer_segments) < 10:
                return self._create_empty_plot("Insufficient data for advanced segmentation")
            
            # Prepare data for clustering
            features = self.customer_segments[['TransactionCount', 'TotalAmount', 'Recency']].dropna()
            
            if len(features) < 10:
                return self._create_empty_plot("Not enough data points for clustering")
            
            # Sample for performance
            if len(features) > 5000:
                features = features.sample(n=2000, random_state=42)
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(features_scaled)
            
            # Add clusters to data
            segmentation_data = features.copy()
            segmentation_data['Cluster'] = clusters
            segmentation_data['ClusterName'] = segmentation_data['Cluster'].map({
                0: 'Low Value', 1: 'Medium Value', 2: 'High Value', 3: 'Premium'
            })
            
            # Create 3D scatter plot
            fig = px.scatter_3d(
                segmentation_data,
                x='TransactionCount',
                y='TotalAmount',
                z='Recency',
                color='ClusterName',
                title='3D Customer Segmentation (K-means Clustering)',
                color_discrete_sequence=[self.colors['primary'], self.colors['secondary'], 
                                       self.colors['accent'], self.colors['warning']]
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=self.colors['text']),
                height=600
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_plot(f"Error in advanced segmentation: {str(e)}")
    
    def create_behavioral_analysis(self):
        """Create behavioral analysis charts"""
        try:
            fig = sp.make_subplots(
                rows=2, cols=2,
                subplot_titles=('Hourly Transaction Pattern', 'Weekly Transaction Pattern',
                              'Transaction Amount Categories', 'Time of Day Analysis'),
                specs=[[{"type": "scatter"}, {"type": "bar"}],
                      [{"type": "pie"}, {"type": "bar"}]]
            )
            
            # Hourly pattern
            hourly_pattern = self.data['TransactionHour'].value_counts().sort_index().reset_index()
            hourly_pattern.columns = ['Hour', 'Count']
            fig.add_trace(
                go.Scatter(x=hourly_pattern['Hour'], y=hourly_pattern['Count'],
                          mode='lines+markers', name='Hourly Pattern',
                          line=dict(color=self.colors['primary'], width=3)),
                row=1, col=1
            )
            
            # Weekly pattern
            weekday_pattern = self.data['TransactionDayOfWeek'].value_counts().sort_index().reset_index()
            weekday_pattern.columns = ['DayOfWeek', 'Count']
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekday_pattern['DayName'] = weekday_pattern['DayOfWeek'].apply(lambda x: days[x] if x < len(days) else 'Unknown')
            fig.add_trace(
                go.Bar(x=weekday_pattern['DayName'], y=weekday_pattern['Count'],
                      name='Weekly Pattern', marker_color=self.colors['secondary']),
                row=1, col=2
            )
            
            # Transaction categories - PIE CHART
            category_counts = self.data['TransactionCategory'].value_counts().head(6)
            fig.add_trace(
                go.Pie(labels=category_counts.index, values=category_counts.values,
                      name='Transaction Categories',
                      marker_colors=[self.colors['primary'], self.colors['secondary'], 
                                   self.colors['accent'], self.colors['warning']]),
                row=2, col=1
            )
            
            # Time of day analysis
            time_of_day = self.data['TimeOfDay'].value_counts()
            fig.add_trace(
                go.Bar(x=time_of_day.index, y=time_of_day.values,
                      name='Time of Day', marker_color=self.colors['accent']),
                row=2, col=2
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=self.colors['text']),
                height=700,
                showlegend=False
            )
            
            return fig
        except Exception as e:
            return self._create_empty_plot(f"Error in behavioral analysis: {str(e)}")
    
    def create_financial_metrics(self):
        """Create comprehensive financial metrics dashboard"""
        try:
            # Calculate key metrics
            total_transactions = len(self.data)
            total_amount = self.data['TransactionAmount (INR)'].sum()
            avg_transaction = self.data['TransactionAmount (INR)'].mean()
            avg_balance = self.data['CustAccountBalance'].mean()
            
            # Top transactions
            top_transactions = self.data.nlargest(5, 'TransactionAmount (INR)')
            top_customers = self.data.groupby('CustomerID')['TransactionAmount (INR)'].sum().nlargest(5)
            
            fig = sp.make_subplots(
                rows=2, cols=2,
                subplot_titles=('Transaction Amount Distribution', 'Account Balance Distribution',
                              'Top 5 Transactions', 'Top 5 Customers by Total Spend'),
                specs=[[{"type": "histogram"}, {"type": "histogram"}],
                      [{"type": "bar"}, {"type": "bar"}]]
            )
            
            # Transaction amount distribution
            fig.add_trace(
                go.Histogram(x=self.data['TransactionAmount (INR)'], 
                            name='Transaction Amount', marker_color=self.colors['primary']),
                row=1, col=1
            )
            
            # Account balance distribution
            fig.add_trace(
                go.Histogram(x=self.data['CustAccountBalance'], 
                            name='Account Balance', marker_color=self.colors['secondary']),
                row=1, col=2
            )
            
            # Top transactions
            fig.add_trace(
                go.Bar(x=top_transactions['TransactionAmount (INR)'], 
                      y=top_transactions['CustomerID'].astype(str) + " - " + top_transactions['CustLocation'],
                      orientation='h', name='Top Transactions',
                      marker_color=self.colors['accent']),
                row=2, col=1
            )
            
            # Top customers
            fig.add_trace(
                go.Bar(x=top_customers.values, y=top_customers.index.astype(str),
                      orientation='h', name='Top Customers',
                      marker_color=self.colors['warning']),
                row=2, col=2
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=self.colors['text']),
                height=700,
                showlegend=False
            )
            
            return fig
        except Exception as e:
            return self._create_empty_plot(f"Error in financial metrics: {str(e)}")
    
    def create_rfm_analysis(self):
        """Create RFM (Recency, Frequency, Monetary) analysis"""
        try:
            if self.customer_segments is None:
                return self._create_empty_plot("No RFM data available")
            
            # Sample for performance
            if len(self.customer_segments) > 5000:
                plot_data = self.customer_segments.sample(n=2000, random_state=42)
            else:
                plot_data = self.customer_segments
            
            fig = sp.make_subplots(
                rows=2, cols=2,
                subplot_titles=('RFM Segment Distribution', 'Recency vs Frequency',
                              'Monetary vs Frequency', 'Customer Value Matrix'),
                specs=[[{"type": "bar"}, {"type": "scatter"}],
                      [{"type": "scatter"}, {"type": "heatmap"}]]
            )
            
            # RFM Segment Distribution
            segment_dist = plot_data.groupby(['FrequencyScore', 'MonetaryScore']).size().reset_index()
            segment_dist['Segment'] = segment_dist['FrequencyScore'] + '-' + segment_dist['MonetaryScore']
            
            fig.add_trace(
                go.Bar(x=segment_dist['Segment'], y=segment_dist[0],
                      name='Segment Distribution', marker_color=self.colors['primary']),
                row=1, col=1
            )
            
            # Recency vs Frequency
            fig.add_trace(
                go.Scatter(x=plot_data['Recency'], y=plot_data['TransactionCount'],
                          mode='markers', name='Recency vs Frequency',
                          marker=dict(color=plot_data['TotalAmount'],
                                    colorscale='Viridis', size=8, showscale=True)),
                row=1, col=2
            )
            
            # Monetary vs Frequency
            fig.add_trace(
                go.Scatter(x=plot_data['TransactionCount'], y=plot_data['TotalAmount'],
                          mode='markers', name='Monetary vs Frequency',
                          marker=dict(color=plot_data['Recency'],
                                    colorscale='Plasma', size=8, showscale=True)),
                row=2, col=1
            )
            
            # Customer Value Matrix - HEATMAP
            rfm_matrix = plot_data.groupby(['FrequencyScore', 'MonetaryScore']).agg({
                'CustomerID': 'count',
                'TotalAmount': 'mean'
            }).reset_index()
            
            heatmap_data = rfm_matrix.pivot(index='FrequencyScore', columns='MonetaryScore', values='CustomerID')
            fig.add_trace(
                go.Heatmap(z=heatmap_data.values, x=heatmap_data.columns, y=heatmap_data.index,
                          colorscale='Purples', name='Customer Count'),
                row=2, col=2
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=self.colors['text']),
                height=700,
                showlegend=False
            )
            
            return fig
        except Exception as e:
            return self._create_empty_plot(f"Error in RFM analysis: {str(e)}")
    
    def create_simple_segmentation(self):
        """Fast customer segmentation for overview"""
        try:
            if self.customer_segments is None or len(self.customer_segments) == 0:
                return self._create_empty_plot("No segmentation data available")
            
            # Sample for performance
            plot_data = self.customer_segments.head(1000)
            
            fig = px.scatter(
                plot_data,
                x='TransactionCount',
                y='TotalAmount',
                color='FrequencyScore',
                size='LastBalance',
                title='Customer Segmentation Overview',
                color_discrete_sequence=[self.colors['primary'], self.colors['secondary'], self.colors['accent']]
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=self.colors['text']),
                height=500
            )
            
            return fig
        except Exception as e:
            return self._create_empty_plot(f"Error in simple segmentation: {str(e)}")
    
    def _create_empty_plot(self, message):
        """Create empty plot with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color=self.colors['text'])
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=300
        )
        return fig
from DataPreprocess import DataPreprocessor
from MMRPredict import MMRPredictor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class VehiclePriceCalculator:
    """
    Calculates whether a vehicle is underpriced or overpriced based on MMR prediction
    and value retention analysis.
    """
    
    def __init__(self, data_path='car_prices.csv', brand_focus=None):
        """
        Initialize the vehicle price calculator.
        
        Parameters:
        -----------
        data_path : str
            Path to the training dataset CSV file
        brand_focus : str
            Specific brand to focus on for analysis
        """
        try:
            self.df = pd.read_csv(data_path)
            print(f"Successfully loaded {data_path} with {len(self.df)} records")
            
            # Identify and print column names for debugging
            print(f"Columns in dataset: {list(self.df.columns)}")
            
        except FileNotFoundError:
            print(f"Error: The file {data_path} was not found.")
            self.df = None
            
        self.brand_focus = brand_focus
        
        if self.df is not None:
            self.preprocessor = DataPreprocessor(self.df)
            self.mmr_predictor = MMRPredictor(self.df, model_type='gradient_boosting')
        
        self.numerical_cols = ['year', 'mileage', 'condition', 'car_age']
        self.categorical_cols = ['brand', 'model', 'body_type', 'transmission']

        self.avg_retention_by_brand = {}
        self.avg_retention_by_model = {}
        self.avg_retention_by_age = {}
        
    def segment(self):
        """
        Segment the data based on brand focus and calculate average value retention.
        
        Returns:
        --------
        self
        """
        # Filter for focus brand if specified
        if self.brand_focus and 'brand' in self.preprocessor.df.columns:
            self.preprocessor.df = self.preprocessor.df[self.preprocessor.df['brand'] == self.brand_focus].copy()
        
        # Calculate average value retention by brand
        if 'brand' in self.preprocessor.df.columns and 'value_retention' in self.preprocessor.df.columns:
            self.avg_retention_by_brand = self.preprocessor.df.groupby('brand')['value_retention'].mean().to_dict()
        
        # Calculate average value retention by model
        if 'market_model' in self.preprocessor.df.columns and 'value_retention' in self.preprocessor.df.columns:
            self.avg_retention_by_model = self.preprocessor.df.groupby('market_model')['value_retention'].mean().to_dict()
        
        # Calculate average value retention by vehicle age
        if 'car_age' in self.preprocessor.df.columns and 'value_retention' in self.preprocessor.df.columns:
            self.avg_retention_by_age = self.preprocessor.df.groupby('car_age')['value_retention'].mean().to_dict()
        
        return self
    
    def prepare_data(self):
        """
        Prepare the data for analysis.
        
        Returns:
        --------
        self
        """
        # Preprocess the data
        self.preprocessor.prepare_data()
        
        # Segment the data based on brand focus
        self.segment()
        
    
    def train_mmr_predictor(self):
        """
        Train the MMR predictor model on the preprocessed data.
        
        Returns:
        --------
        self
        """
        self.preprocessor.prepare_data()
        self.mmr_predictor.preprocess_data()
        self.mmr_predictor.train_evaluate_optimize()
        
        return self.mmr_predictor.model
    
    def prepare_vehicle_for_prediction(self, vehicle_df):
        """
        Prepare a single vehicle's data for MMR prediction.
        
        Parameters:
        -----------
        vehicle_df : pandas DataFrame
            Single vehicle data
            
        Returns:
        --------
        pandas DataFrame
            Processed vehicle data ready for prediction
        """
        print("Preparing vehicle for MMR prediction...")
        
        # Create a copy to avoid modifying the original
        processed_df = vehicle_df.copy()
        
        # Ensure all required columns exist (add with default values if missing)
        for col in self.numerical_cols:
            if col not in processed_df.columns:
                if col == 'car_age' and 'year' in processed_df.columns:
                    current_year = pd.Timestamp.now().year
                    processed_df['car_age'] = current_year - processed_df['year'].iloc[0]
                    print(f"Created 'car_age' with value: {processed_df['car_age'].iloc[0]}")
                else:
                    processed_df[col] = 0
                    print(f"Added missing column '{col}' with default value 0")
        
        # Handle categorical columns
        for col in self.categorical_cols:
            if col not in processed_df.columns:
                processed_df[col] = 'unknown'
                # print(f"Added missing column '{col}' with default value 'unknown'")
        
        # Create the market_model column if it doesn't exist but model does
        if 'market_model' not in processed_df.columns and 'model' in processed_df.columns:
            processed_df['market_model'] = 'Other'
            model_value = processed_df['model'].iloc[0]
            
            # Map model to market category if possible
            for category, models in self.market_map.items():
                if model_value in models:
                    processed_df['market_model'] = category
                    print(f"Mapped model '{model_value}' to market category '{category}'")
                    break
        
        # Convert categorical columns to numeric using LabelEncoder
        for col in self.categorical_cols:
            if col in processed_df.columns:
                # For simplicity, just use 0 for prediction purposes
                # In a production system, you'd want to use the same encoding as during training
                processed_df[col] = 0
        
        # Ensure all columns are numeric
        for col in processed_df.columns:
            if processed_df[col].dtype == 'object':
                print(f"Converting column '{col}' from object to numeric")
                processed_df[col] = 0
        
        # Select only the columns used by the model
        model_columns = self.numerical_cols + self.categorical_cols
        available_columns = [col for col in model_columns if col in processed_df.columns]
        
        print(f"Using columns for prediction: {available_columns}")
        result_df = processed_df[available_columns]
        
        return result_df
    
    def analyze_vehicle(self, vehicle_data, mmr=None, selling_price=None):
        """
        Analyze a single vehicle to determine if it's underpriced or overpriced.
        """

        # Convert dict to DataFrame if necessary
        if isinstance(vehicle_data, dict):
            vehicle_df = pd.DataFrame([vehicle_data])
        else:
            vehicle_df = vehicle_data.copy()
        
        # print(f"Analyzing vehicle with columns: {vehicle_df.columns.tolist()}")
        
        # Add MMR if provided
        if mmr is not None:
            vehicle_df['mmr'] = mmr
            
        # Standardize selling price column name
        if selling_price is not None:
            vehicle_df['selling_price'] = selling_price
        
        # Handle both potential column names
        if 'selling_price' in vehicle_df.columns and 'sellingprice' not in vehicle_df.columns:
            vehicle_df['sellingprice'] = vehicle_df['selling_price']
        elif 'sellingprice' in vehicle_df.columns and 'selling_price' not in vehicle_df.columns:
            vehicle_df['selling_price'] = vehicle_df['sellingprice']
        
        # Predict MMR if not provided
        if mmr is None and 'mmr' not in vehicle_df.columns:
            if self.mmr_predictor.model is None:
                print("Error: MMR predictor model has not been trained.")
                return None
                
            # try:
            # 
            vehicle_preprocessor = DataPreprocessor(vehicle_df)
            vehicle_preprocessor.prepare_data()
            predicted_mmr = self.mmr_predictor.model.predict(vehicle_df)
            # Predict MMR
            # predicted_mmr = self.mmr_predictor.model.predict(X_pred)[0]
            vehicle_df['mmr'] = predicted_mmr
            print(f"Predicted MMR: ${predicted_mmr:.2f}")
            # except Exception as e:
            #     print(f"Error predicting MMR: {e}")
            #     # For demonstration purposes, use a default MMR value
            #     vehicle_df['mmr'] = 15000
            #     print("Using default MMR value for demonstration: $15,000")
        
        # Calculate value retention
        if 'selling_price' in vehicle_df.columns and 'mmr' in vehicle_df.columns:
            value_retention = vehicle_df['selling_price'].iloc[0] / vehicle_df['mmr'].iloc[0]
        else:
            value_retention = None
        
        # Get model and brand information
        brand = vehicle_df['brand'].iloc[0] if 'brand' in vehicle_df.columns else None
        model = vehicle_df['model'].iloc[0] if 'model' in vehicle_df.columns else None
        car_age = vehicle_df['car_age'].iloc[0] if 'car_age' in vehicle_df.columns else None
        
        # Get average retention metrics for comparison (using defaults if not available)
        model_avg_retention = self.avg_retention_by_model.get(model, 1.0)
        brand_avg_retention = self.avg_retention_by_brand.get(brand, 1.0)
        age_avg_retention = self.avg_retention_by_age.get(car_age, 1.0)
        
        # Determine price status
        if value_retention is not None:
            if value_retention < 0.9:
                price_status = 'underpriced'
                price_deviation = (0.9 - value_retention) * 100
            elif value_retention > 1.1:
                price_status = 'overpriced'
                price_deviation = (value_retention - 1.1) * 100
            else:
                price_status = 'fair_price'
                price_deviation = 0
        else:
            price_status = None
            price_deviation = None
        
        # Calculate recommended price range
        if 'mmr' in vehicle_df.columns:
            mmr_value = vehicle_df['mmr'].iloc[0]
            min_fair_price = 0.9 * mmr_value
            max_fair_price = 1.1 * mmr_value
        else:
            min_fair_price = None
            max_fair_price = None
        
        # Return analysis results
        return {
            'vehicle_details': vehicle_df.iloc[0].to_dict(),
            'mmr': vehicle_df['mmr'].iloc[0] if 'mmr' in vehicle_df.columns else None,
            'selling_price': vehicle_df['selling_price'].iloc[0] if 'selling_price' in vehicle_df.columns else None,
            'value_retention': value_retention,
            'price_status': price_status,
            'price_deviation': price_deviation,
            'model_avg_retention': model_avg_retention,
            'brand_avg_retention': brand_avg_retention,
            'age_avg_retention': age_avg_retention,
            'min_fair_price': min_fair_price,
            'max_fair_price': max_fair_price
        }

    def analyze_inventory(self, inventory_df):
        """
        Analyze an entire inventory to identify underpriced and overpriced vehicles.
        
        Parameters:
        -----------
        inventory_df : pandas DataFrame
            Vehicle inventory data
            
        Returns:
        --------
        pandas DataFrame
            Inventory with price analysis
        """
        # Create a copy to avoid modifying the original DataFrame
        analyzed_inventory = inventory_df.copy()
        
        # Process each vehicle in the inventory
        results = []
        for i, row in analyzed_inventory.iterrows():
            vehicle_data = row.to_dict()
            analysis = self.analyze_vehicle(vehicle_data)
            results.append(analysis)
        
        # Create a DataFrame from the analysis results
        result_df = pd.DataFrame(results)
        
        # Add columns for value retention and price status
        analyzed_inventory['predicted_mmr'] = result_df['mmr']
        analyzed_inventory['value_retention'] = result_df['value_retention']
        analyzed_inventory['price_status'] = result_df['price_status']
        analyzed_inventory['min_fair_price'] = result_df['min_fair_price']
        analyzed_inventory['max_fair_price'] = result_df['max_fair_price']
        
        return analyzed_inventory
    
    def plot_churn_analysis_heatmap(self, df, segment=None):
        """
        Generate a heatmap of value retention by model year and age.
        
        Parameters:
        -----------
        df : pandas DataFrame
            Vehicle data with value retention information
        segment : str
            Filter by vehicle segment
            
        Returns:
        --------
        matplotlib figure
            Value retention heatmap
        """
        # Filter data if segment is specified
        plot_df = df.copy()
        # if segment:
        #     plot_df = plot_df[plot_df['segment'] == segment]
        
        # Create pivot table of value retention by year and age
        pivot = plot_df.pivot_table(
            values='value_retention',
            index='year',
            columns='car_age',
            aggfunc='mean'
        )
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt='.2f', cbar_kws={'label': 'Value Retention'})
        plt.title(f'Value Retention by Year and Age{" for " + segment if segment else ""}')
        plt.xlabel('Vehicle Age (Years)')
        plt.ylabel('Model Year')
        
        plt.savefig('dataviz/churn_analysis_heatmap.png')
        plt.close()
    
    def plot_price_status_distribution(self, df):
        """
        Generate a bar chart showing the distribution of price statuses.
        
        Parameters:
        -----------
        df : pandas DataFrame
            Vehicle data with price status information
            
        Returns:
        --------
        matplotlib figure
            Price status distribution chart
        """
        # Count vehicles by price status
        status_counts = df['price_status'].value_counts()
        
        # Create bar chart
        plt.figure(figsize=(10, 6))
        sns.barplot(x=status_counts.index, y=status_counts.values)
        plt.title('Distribution of Price Statuses in Inventory')
        plt.xlabel('Price Status')
        plt.ylabel('Count')
        plt.xticks(rotation=45)

        plt.savefig('dataviz/price_status_distribution.png')
        plt.close()

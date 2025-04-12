 
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



class DataPreprocessor:
    """
    Handles data cleaning (including data visualization), feature engineering, and preprocessing for mmr analysis.
    """
    def __init__(self, data):
        """
        Initialize the DataPreprocessor object, read the file, and clean the data.
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file to be read and cleaned
        """
        # Read the file
        self.df = data
        self.numerical_cols = ['condition', 'car_age', 'mileage']
        self.categorical_cols = ['brand', 'market_model', 'body_type', 'transmission', 'state', 'interior', 'color']
        self.market_map = {
        "Economy Sedan": [
            "Altima", "Focus", "Impala", "Sonata", "Cruze", "Taurus",
            "Optima", "200", "Avenger", "Passat", "Civic", "Corolla",
            "Fusion", "Malibu", "Sentra", "Elantra", "Jetta", "Accord",
            "Camry", "Versa", "Rio", "Yaris", "Forte", "Fiesta", "Sonic", "Cobalt", "Sebring"],
        
        "Luxury Sedan": [
            "3 Series", "5 Series", "7 Series", "A4", "A6", "A8", "C-Class", 
            "E-Class", "S-Class", "S60", "S90", "TLX", "Q50", "Genesis",
            "Maxima", "300", "G Sedan", "LS", "ES", "IS"],
        
        "Sports Sedan": [
            "M3", "M5", "Charger", "S4", "CTS-V", "IS F", "XFR", "WRX"],
        
        "Economy SUV": [
            "Explorer", "Edge", "Journey", "Escape", "Rogue", "Tucson",
            "Equinox", "Sorento", "CX-5", "CR-V", "RAV4", "Highlander",
            "Santa Fe", "Kicks", "HR-V", "Sportage", "Soul", "Pilot", "Pathfinder", "Traverse", "Durango"],
        
        "Luxury SUV": [
            "Grand Cherokee", "X5", "X3", "Q7", "Q5", "GLC", "GLA", 
            "RX", "LX", "Cayenne", "MDX", "Escalade", "Navigator",
            "GLE", "Macan", "XC90", "Tahoe", "Suburban", "Murano", "Expedition"],
        
        "Off-Road SUV": [
            "Wrangler", "4Runner", "Bronco", "Defender", "G-Class"],
        
        "Pickup Truck": [
            "1500", "F-150", "Silverado 1500", "Ram Pickup 1500", 
            "Tacoma", "Tundra", "Ranger", "Colorado", "Frontier", "F-250 Super Duty", "Silverado 2500HD"],
        
        "Electric Vehicle": [
            "Leaf", "Model S", "Model 3", "Model X", "Model Y",
            "Bolt", "i3", "i4", "Polestar 2", "Mach-E"],
        
        "Sports Car": [
            "Mustang", "Camaro", "Corvette", "370Z", "911", "M4",
            "Supra", "GT-R", "F-Type", "718 Cayman"],
        
        "Minivan": [
            "Sienna", "Odyssey", "Grand Caravan", "Town and Country",
            "Quest", "Pacifica", "Sedona"],
        
        "Hybrid Car": [
            "Prius", "Camry Hybrid", "Accord Hybrid", "Fusion Hybrid",
            "Highlander Hybrid", "RAV4 Hybrid"],
        
        "Economy Hatchback": [
            "Caliber", "PT Cruiser", "Accent"],
    }

    # Create a function to read the file from specified path (if needed)
    def read_file(self, file_path):
        """
        Read a CSV file from the specified path.
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file
            
        Returns:
        --------
        pandas DataFrame
            Data read from the CSV file
        """
        return pd.read_csv(file_path)

    def clean_data(self):
        """
        Clean raw vehicle data by handling missing values and extreme values.
        
        Returns:
        --------
        pandas DataFrame
            Cleaned vehicle data
        """
        # First print column names for debugging
        print(f"Columns before cleaning: {self.df.columns.tolist()}")
        
        # Handle missing values and duplicates
        self.df.dropna(how='any', inplace=True)
        self.df.drop_duplicates(inplace=True)

        # Rename some columns for better readability - only if they exist
        rename_map = {'make': 'brand', 'body': 'body_type', 'odometer': 'mileage'}
        columns_to_rename = {old: new for old, new in rename_map.items() if old in self.df.columns}
        if columns_to_rename:
            self.df.rename(columns=columns_to_rename, inplace=True)

        # Drop unnecessary columns only if they exist
        columns_to_drop = ['vin']
        existing_columns_to_drop = [col for col in columns_to_drop if col in self.df.columns]
        if existing_columns_to_drop:
            self.df.drop(columns=existing_columns_to_drop, inplace=True)

        # Calculate sales year based on the sales date
        if 'saledate' in self.df.columns:
            # Extract year from saledate - handling different possible formats
            try:
                # First try to extract from standard format
                self.df['sale_year'] = self.df['saledate'].str[11:15]
                self.df['sale_year'] = pd.to_numeric(self.df['sale_year'], errors='coerce')
                
                # If we got NaN values, try parsing as datetime
                if self.df['sale_year'].isna().any():
                    self.df['sale_year'] = pd.to_datetime(self.df['saledate'], errors='coerce').dt.year
                    
                print(f"Extracted sale years from {self.df['sale_year'].min()} to {self.df['sale_year'].max()}")
            except Exception as e:
                print(f"Warning: Could not extract year from saledate: {e}")
                # If year extraction fails, use current year as fallback
                current_year = pd.Timestamp.now().year
                self.df['sale_year'] = current_year
                print(f"Using current year ({current_year}) as fallback for sale_year")

        # Handle extreme values (outliers)
        # Filter out extreme mmr values based on p-values
        if 'mmr' in self.df.columns:
            mmr_quantiles = self.df['mmr'].quantile([0.02, 0.98])
            lower_bound, upper_bound = mmr_quantiles[0.02], mmr_quantiles[0.98]
            print(f"MMR bounds: {lower_bound} to {upper_bound}")
            self.df = self.df[(self.df['mmr'] >= lower_bound) & (self.df['mmr'] <= upper_bound)]
                
        # Filter out extreme mileage values based on p-values
        if 'mileage' in self.df.columns:
            mileage_quantiles = self.df['mileage'].quantile([0.02, 0.98])
            lower_bound, upper_bound = mileage_quantiles[0.02], mileage_quantiles[0.98]
            print(f"Mileage bounds: {lower_bound} to {upper_bound}")
            self.df = self.df[(self.df['mileage'] >= lower_bound) & (self.df['mileage'] <= upper_bound)]
                
        # Filter out extreme price values based on p-values
        price_col = 'sellingprice' if 'sellingprice' in self.df.columns else 'selling_price'
        if price_col in self.df.columns:
            price_quantiles = self.df[price_col].quantile([0.02, 0.98])
            lower_bound, upper_bound = price_quantiles[0.02], price_quantiles[0.98]
            print(f"Price bounds: {lower_bound} to {upper_bound}")
            self.df = self.df[(self.df[price_col] >= lower_bound) & (self.df[price_col] <= upper_bound)]
                
        # Filter out unrealistic years (e.g., after manufacturer year and before future years)
        if all(col in self.df.columns for col in ['sale_year', 'year']):
            current_year = pd.Timestamp.now().year
            self.df = self.df[self.df['sale_year'] >= self.df['year']]
            self.df = self.df[self.df['sale_year'] <= current_year]
            print(f"Filtered years to keep records where sale_year is between vehicle year and current year")

        # Calculate car age if we have both year and sale_year
        if all(col in self.df.columns for col in ['year', 'sale_year']):
            self.df['car_age'] = self.df['sale_year'] - self.df['year']
            print(f"Created car_age column with range: {self.df['car_age'].min()} to {self.df['car_age'].max()}")
        elif 'year' in self.df.columns:
            # If we don't have sale_year, use current year to calculate car_age
            current_year = pd.Timestamp.now().year
            self.df['car_age'] = current_year - self.df['year']
            print(f"Created car_age column using current year: {current_year}")
            
        print(f"Data shape after cleaning: {self.df.shape}")
        print(f"Columns after cleaning: {self.df.columns.tolist()}")
        

    def generate_numerical_histograms(self):
        """
        Generate histograms for numerical columns and save them to a folder.
        """

        if not os.path.exists('dataviz'):
            os.makedirs('dataviz')
        # generate histogram for numerical columns and output it to a folder called dataviz
        for col in self.numerical_cols:
            if col in self.df.columns:
                self.df[col].hist(bins=30)
                plt.title(f'Histogram of {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                plt.savefig(f'dataviz/{col}_histogram.png')
                plt.close()
    
    def map_market_category(self):
        """
        Map vehicle brands to market categories.
        """
        # Check if required columns exist
        if 'brand' not in self.df.columns:
            print("Warning: Cannot map market categories - 'brand' column is missing")
            return
        
        # Remap the brand column so that only top 10 brands are kept, rest are renamed to 'Other'
        top_brands = self.df['brand'].value_counts().nlargest(10).index
        self.df['brand'] = self.df['brand'].where(self.df['brand'].isin(top_brands), 'Other')

        # Create or modify market_model column
        if 'model' in self.df.columns:
            # Create a new column for brand model if it doesn't exist
            if 'market_model' not in self.df.columns:
                self.df['market_model'] = 'Other'
            
            # Map each model to its corresponding market category
            for category, models in self.market_map.items():
                self.df.loc[self.df['model'].isin(models), 'market_model'] = category
                
            print(f"Mapped models to {self.df['market_model'].nunique()} market categories")
        else:
            print("Warning: 'model' column not found. Cannot map to market categories.")

    def generate_market_category_histogram(self):
        """
        Generate histogram for market categories and save them to a folder.
        """
        if not os.path.exists('dataviz'):
            os.makedirs('dataviz')
        # generate histogram for market categories and output it to a folder called dataviz
        self.df['market_model'].value_counts().plot(kind='bar')
        plt.title('Market Category Distribution')
        plt.xlabel('Market Category')
        plt.ylabel('Frequency')
        plt.savefig('dataviz/market_category_histogram.png')
        plt.close()

    def calculate_value_retention(self):
        """
        Calculate value retention and price status based on selling price and MMR.
        """
        if 'sellingprice' in self.df.columns and 'mmr' in self.df.columns:
            self.df['value_retention'] = (self.df['sellingprice'] - self.df['mmr']) / self.df['mmr']
            self.df['price_status'] = pd.cut(self.df['value_retention'], bins=[-float('inf'), -0.1, 0.1, float('inf')], labels=['Underpriced', 'Fair', 'Overpriced'])
    
    def generate_mmr_distribution(self):
        '''
        Generate scatter plots and box plots to visualize the relationship between features and MMR.
        '''
        if not os.path.exists('dataviz'):
            os.makedirs('dataviz')
        num_features = ['mileage', 'condition', 'car_age']
        for feature in num_features:
            plt.figure(figsize=(8, 5))
            sns.scatterplot(x=self.df[feature], y=self.df['mmr'], alpha=0.3)
            plt.xlabel(feature)
            plt.ylabel("MMR")
            plt.title(f"MMR vs {feature}")
            plt.savefig(f'dataviz/mmr_distribution_by_{feature}.png')
            plt.close()
            
        # Box plot: show the impact of categorical variables on mmr
        cat_features = ['market_model', 'brand', 'transmission']
        for feature in cat_features:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x=self.df[feature], y=self.df['mmr'], data=self.df)
            plt.xticks(rotation=45)
            plt.xlabel(feature)
            plt.ylabel("MMR")
            plt.title(f"MMR Distribution by {feature}")
            plt.savefig(f'dataviz/mmr_distribution_by_{feature}.png')
            plt.close()

    def generate_price_distribution(self):
        '''
        Generate scatter plots and box plots to visualize the relationship between features and selling price.
        '''
        if not os.path.exists('dataviz'):
            os.makedirs('dataviz')
        num_features = ['mileage', 'condition', 'car_age']
        for feature in num_features:
            plt.figure(figsize=(8, 5))
            sns.scatterplot(x=self.df[feature], y=self.df['sellingprice'], alpha=0.3)
            plt.xlabel(feature)
            plt.ylabel("Selling Price")
            plt.title(f"Selling Price vs {feature}")
            plt.savefig(f'dataviz/price_distribution_by_{feature}.png')
            plt.close()
            
        # Box plot: show the impact of categorical variables on mmr
        cat_features = ['market_model', 'brand', 'transmission']
        for feature in cat_features:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x=self.df[feature], y=self.df['sellingprice'], data=self.df)
            plt.xticks(rotation=45)
            plt.xlabel(feature)
            plt.ylabel("Selling Price")
            plt.title(f"Price Distribution by {feature}")
            plt.savefig(f'dataviz/price_distribution_by_{feature}.png')
            plt.close()


    def prepare_data(self):
        """
        Full data preprocessing pipeline including cleaning, feature engineering,
        and encoding categorical features.
        
        Parameters:
        -----------
        df : pandas DataFrame
            Raw vehicle data
        training : bool
            If True, fit and transform the data; if False, only transform
            
        Returns:
        --------
        pandas DataFrame
            Fully preprocessed vehicle data ready for modeling
        """
        # Apply each preprocessing step
        self.clean_data()
        # Generate histograms for numerical columns
        self.generate_numerical_histograms()
        # Map vehicle brands to market categories
        self.map_market_category()
        # Generate histogram for market categories
        self.generate_market_category_histogram()
        # Calculate value retention
        self.calculate_value_retention()
        # Generate plots for MMR distribution
        self.generate_mmr_distribution()
        # Generate plots for price distribution
        self.generate_price_distribution()
        
    
    # Output a csv file of the preprocessed data
    def output_csv(self, file_name):
        """
        Output the preprocessed DataFrame to a CSV file.
        
        Parameters:
        -----------
        df : pandas DataFrame
            Preprocessed vehicle data
        file_name : str
            Name of the output CSV file
        """

        self.df.to_csv(file_name, index=False)
        
# write main to read the file and call the functions
if __name__ == "__main__":
    # Example usage
    data = pd.read_csv('car_prices.csv')

    # Create an instance of the DataPreprocessor class
    preprocessor = DataPreprocessor(data)
    
    # Preprocess the data
    preprocessor.prepare_data()
    
    # Output the preprocessed data to a CSV file (this is the output for this module)
    preprocessor.output_csv('preprocessed_data.csv')

    # Display the first few rows of the preprocessed data
    print(preprocessor.df.head())
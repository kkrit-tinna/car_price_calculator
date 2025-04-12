from PriceCalculate import VehiclePriceCalculator
import pandas as pd

if __name__ == "__main__":
    # Specify the path to your training data
    training_data_path = 'car_prices.csv'  # Replace with your actual training file
    
    # Create a price calculator for a specific brand (optional)
    calculator = VehiclePriceCalculator(data_path=training_data_path, brand_focus=None)  # Remove brand focus if causing issues
    
    if calculator.df is None:
        print("Error: Could not load training data.")
        exit(1)
    
    # # Check if mmr column exists
    # if 'mmr' not in calculator.df.columns:
    #     print(f"Error: Required 'mmr' column not found in the data. Available columns: {calculator.df.columns.tolist()}")
    #     exit(1)
    
    # Display data summary
    print("\nData summary:")
    print(calculator.df.describe().transpose())
    
    # Print info about mmr column specifically
    print("\nMMR column info:")
    print(f"MMR column type: {calculator.df['mmr'].dtype}")
    print(f"MMR column range: {calculator.df['mmr'].min()} to {calculator.df['mmr'].max()}")
    print(f"MMR missing values: {calculator.df['mmr'].isna().sum()}")
    
    # Prepare the data
    print("\nPreparing data...")
    calculator.preprocessor.prepare_data()
    calculator.segment()
    
    # Train the MMR predictor model
    print("\nPreprocessing and training model...")
    success = False
    try:
        calculator.mmr_predictor.preprocess_data()
        success = calculator.mmr_predictor.train_evaluate_optimize()
    except Exception as e:
        print(f"Error during model training: {e}")
    
    if not success:
        print("Model training failed. Cannot proceed with analysis.")
        exit(1)
    
    # Modified test vehicle to match typical model structure
    vehicle = {
        'brand': 'Toyota',
        'model': 'Camry',
        'year': 2018,
        'mileage': 45000,
        'condition': 4,  # Using numeric value for condition (1-5 scale)
        'body_type': 'Sedan',
        'transmission': 'Automatic',
        'selling_price': 18500
    }
    
    # Basic functionality test - even if model training fails
    analysis = calculator.analyze_vehicle(vehicle)
    
    if analysis:
        print("\nVehicle Analysis:")
        print(f"MMR: ${analysis['mmr']:.2f}")
        print(f"Selling Price: ${analysis['selling_price']:.2f}")
        print(f"Value Retention: {analysis['value_retention']:.2f}")
        print(f"Price Status: {analysis['price_status']}")
        print(f"Recommended Price Range: ${analysis['min_fair_price']:.2f} - ${analysis['max_fair_price']:.2f}")
    
    # Analyze entire inventory (for example)
    inventory = pd.read_csv('vehicle_data.csv') # this is an example dataset
    
    analyzed_inventory = calculator.analyze_inventory(inventory)
    # print out the first few rows of the analyzed inventory
    # print("\nAnalyzed Inventory:")
    # print(analyzed_inventory.head())
    
    # Count vehicles by price status
    price_status_counts = analyzed_inventory['price_status'].value_counts()
    print("\nInventory Analysis:")
    print(price_status_counts)


    # Plot value retention heatmap ***NEED HELP ON THIS!***
    # calculator.plot_churn_analysis_heatmap(inventory)
    
    # Plot price status distribution
    calculator.plot_price_status_distribution(analyzed_inventory)
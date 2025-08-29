import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def main():
    print("🚀 HOUSE PRICE PREDICTOR - LINEAR REGRESSION PROJECT")
    print("=" * 60)
    
    # Create sample data
    print("🏠 Creating sample house data...")
    np.random.seed(42)
    
    sizes = np.random.uniform(800, 3000, 100)
    prices = 100000 + (sizes * 200) + np.random.normal(0, 25000, 100)
    
    data = pd.DataFrame({
        'size_sqft': sizes,
        'price_dollars': prices
    })
    
    print(f"✅ Created data with {len(data)} houses")
    print(f"   Size range: {sizes.min():.0f} - {sizes.max():.0f} sq ft")
    print(f"   Price range: ${prices.min():,.0f} - ${prices.max():,.0f}")
    
    # Prepare data
    print("\n🔧 Preparing data for training...")
    X = data[['size_sqft']].values
    y = data['price_dollars'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"✅ Training set: {len(X_train)} houses")
    print(f"✅ Testing set: {len(X_test)} houses")
    
    # Train model
    print("\n🤖 Training the model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    slope = model.coef_[0]
    intercept = model.intercept_
    print("✅ Model trained successfully!")
    print(f"   Equation: Price = ${slope:.2f} × Size + ${intercept:,.2f}")
    
    # Evaluate model
    print("\n📈 Evaluating model performance...")
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"✅ R² Score: {r2:.4f}")
    
    if r2 > 0.8:
        print("   🎉 Excellent fit!")
    
    # Make predictions
    print("\n🔮 Making predictions for new houses...")
    new_sizes = [1000, 1500, 2000, 2500, 3000]
    print("Size (sq ft) | Predicted Price")
    print("-" * 35)
    
    for size in new_sizes:
        prediction = model.predict([[size]])[0]
        print(f"{size:>10} | ${prediction:>12,.0f}")
    
    print("\n🎉 PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main() 
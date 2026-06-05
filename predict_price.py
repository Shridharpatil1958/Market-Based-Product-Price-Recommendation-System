import joblib
import pandas as pd

model = joblib.load('../models/price_model.pkl')

def predict_price(product_data):
    df = pd.DataFrame([product_data])
    prediction = model.predict(df)
    return prediction[0]

if __name__ == "__main__":
    sample = {
        "Demand": 150,
        "Competitor_Price": 1200,
        "Rating": 4.5
    }

    price = predict_price(sample)
    print("Recommended Price:", round(price, 2))

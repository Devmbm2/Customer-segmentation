from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
from sklearn.preprocessing import StandardScaler
import asyncio
from sklearn.cluster import KMeans
import pandas as pd

# Load the trained KMeans model and scaler
kmeans = joblib.load('../saved_models/kmeans_model.pkl')
scaler = joblib.load('../saved_models/scaler.pkl')  # Assuming you saved the scaler used for feature scaling

# Initialize FastAPI app
app = FastAPI()

# Example model input format for customer data
class Customer(BaseModel):
    age: float
    transaction_count: int
    total_amount: float
    average_amount: float
    recency: float

# Function to clean and limit text (if needed, based on the customer's inputs)
def clean_text(text: str, max_length: int = 500):
    # Remove unwanted characters (e.g., special characters, numbers, etc.)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Limit the text length
    text = text[:max_length]  # Limit the text to max_length characters
    
    return text

# Define a prediction endpoint for customer segmentation
@app.post("/predict/")
async def predict(customer: Customer):
    # Simulate an asynchronous delay (e.g., time for preprocessing or model inference)
    await asyncio.sleep(1)
    
    # Create a DataFrame from the incoming customer data
    customer_data = pd.DataFrame([{
        'Age': customer.age,
        'TransactionCount': customer.transaction_count,
        'TotalAmount': customer.total_amount,
        'AverageAmount': customer.average_amount,
        'Recency': customer.recency
    }])
    
    # Scale the features using the scaler
    scaled_features = scaler.transform(customer_data)
    
    # Predict the cluster for this customer using the trained KMeans model
    cluster_prediction = kmeans.predict(scaled_features)
    
    # Get the cluster label based on the prediction
    cluster_labels = {
        0: 'Young High Spenders',
        1: 'Senior Regulars',
        2: 'Occasional Low Spenders',
        3: 'Urban Frequent Shoppers'
    }
    
    # Return the predicted segment for the customer
    return {"predicted_segment": cluster_labels[cluster_prediction[0]]}

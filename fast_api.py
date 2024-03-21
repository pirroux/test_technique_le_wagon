from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import csv
import pickle

app = FastAPI()

REQUIRED_COLUMNS = ['uuid', 'account_amount_added_12_24m',
       'account_incoming_debt_vs_paid_0_24m', 'account_status',
       'account_worst_status_0_3m', 'account_worst_status_12_24m',
       'account_worst_status_3_6m', 'account_worst_status_6_12m', 'age',
       'avg_payment_span_0_12m', 'avg_payment_span_0_3m', 'merchant_group',
       'has_paid', 'max_paid_inv_0_12m', 'max_paid_inv_0_24m',
       'num_active_div_by_paid_inv_0_12m', 'num_active_inv',
       'num_arch_ok_0_12m', 'num_arch_ok_12_24m', 'num_unpaid_bills',
       'status_last_archived_0_24m', 'status_2nd_last_archived_0_24m',
       'status_3rd_last_archived_0_24m', 'status_max_archived_0_6_months',
       'status_max_archived_0_12_months', 'status_max_archived_0_24_months',
       'sum_capital_paid_account_0_12m', 'sum_capital_paid_account_12_24m',
       'sum_paid_inv_0_12m', 'time_hours', 'worst_status_active_inv']

@app.get("/")
def root():
    return {
    'greeting': 'Welcome to default_paymentts API!'
}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded CSV file with pandas
    new_data = pd.read_csv(file.file)

    # Check if the CSV file contains all the required columns
    if not set(REQUIRED_COLUMNS).issubset(df.columns):
        raise HTTPException(status_code=400, detail="Invalid CSV file. Required columns are missing.")

    # Apply your trained model to the data
    new_data_pred = new_data.drop(columns="uuid")
    #import model
    with open('model_pipeline.pkl', 'rb') as file:
        pipeline = pickle.load(file)

    # Replace this with your model's prediction method
    predictions = pipeline.predict_proba(new_data_pred)
    # Convert the prediction result back to CSV
    prediction_csv = predictions.to_csv(index=False)

    # Return the CSV data as a string
    return {"prediction": prediction_csv}

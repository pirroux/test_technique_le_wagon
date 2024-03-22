from fastapi import FastAPI
import pandas as pd
import pickle
import sklearn
import requests
import json

app = FastAPI()

# preloading of the model
#app.state.model = pickle.load(open('model.pkl', 'rb'))


@app.get("/")
def root():
    return {"message": "Hello wesh"}

@app.get('/predict')
def predict(uuid, account_amount_added_12_24m, account_days_in_dc_12_24m,
    account_days_in_rem_12_24m, account_days_in_term_12_24m, age,
    avg_payment_span_0_12m, merchant_category, merchant_group,
    has_paid, max_paid_inv_0_12m, max_paid_inv_0_24m, name_in_email,
    num_active_div_by_paid_inv_0_12m, num_active_inv,
    num_arch_dc_0_12m, num_arch_dc_12_24m, num_arch_ok_0_12m,
    num_arch_ok_12_24m, num_arch_rem_0_12m,
    num_arch_written_off_0_12m, num_arch_written_off_12_24m,
    num_unpaid_bills, status_last_archived_0_24m,
    status_2nd_last_archived_0_24m, status_3rd_last_archived_0_24m,
    status_max_archived_0_6_months, status_max_archived_0_12_months,
    status_max_archived_0_24_months, recovery_debt,
    sum_capital_paid_account_0_12m, sum_capital_paid_account_12_24m,
    sum_paid_inv_0_12m, time_hours):

    # Retrieve the data from the request and put it into a dataframe
    df = pd.DataFrame({
        'uuid': [uuid],
        'account_amount_added_12_24m': [account_amount_added_12_24m],
        'account_days_in_dc_12_24m': [account_days_in_dc_12_24m],
        'account_days_in_rem_12_24m': [account_days_in_rem_12_24m],
        'account_days_in_term_12_24m': [account_days_in_term_12_24m],
        'age': [age],
        'avg_payment_span_0_12m': [avg_payment_span_0_12m],
        'merchant_category': [merchant_category],
        'merchant_group': [merchant_group],
        'has_paid': [has_paid],
        'max_paid_inv_0_12m': [max_paid_inv_0_12m],
        'max_paid_inv_0_24m': [max_paid_inv_0_24m],
        'name_in_email': [name_in_email],
        'num_active_div_by_paid_inv_0_12m': [num_active_div_by_paid_inv_0_12m],
        'num_active_inv': [num_active_inv],
        'num_arch_dc_0_12m': [num_arch_dc_0_12m],
        'num_arch_dc_12_24m': [num_arch_dc_12_24m],
        'num_arch_ok_0_12m': [num_arch_ok_0_12m],
        'num_arch_ok_12_24m': [num_arch_ok_12_24m],
        'num_arch_rem_0_12m': [num_arch_rem_0_12m],
        'num_arch_written_off_0_12m': [num_arch_written_off_0_12m],
        'num_arch_written_off_12_24m': [num_arch_written_off_12_24m],
        'num_unpaid_bills': [num_unpaid_bills],
        'status_last_archived_0_24m': [status_last_archived_0_24m],
        'status_2nd_last_archived_0_24m': [status_2nd_last_archived_0_24m],
        'status_3rd_last_archived_0_24m': [status_3rd_last_archived_0_24m],
        'status_max_archived_0_6_months': [status_max_archived_0_6_months],
        'status_max_archived_0_12_months': [status_max_archived_0_12_months],
        'status_max_archived_0_24_months': [status_max_archived_0_24_months],
        'recovery_debt': [recovery_debt],
        'sum_capital_paid_account_0_12m': [sum_capital_paid_account_0_12m],
        'sum_capital_paid_account_12_24m': [sum_capital_paid_account_12_24m],
        'sum_paid_inv_0_12m': [sum_paid_inv_0_12m],
        'time_hours': [time_hours]
    })

    # drop the same columns as in exploration
    # df.drop(columns=['worst_status_active_inv',
    #                 'account_worst_status_12_24m',
    #                 'account_worst_status_6_12m',
    #                 'account_incoming_debt_vs_paid_0_24m',
    #                 'account_worst_status_3_6m',
    #                 'account_status',
    #                 'account_worst_status_0_3m',
    #                 'avg_payment_span_0_3m'], inplace=True)

    # load the model
    new_pipe = pickle.load(open('model.pkl', 'rb'))

    # predict
    return new_pipe.predict_proba(df)

#test the predict API a la mano

#http://127.0.0.1:8000/predict?uuid=19689e3e-b3a1-4339-987b-ac1b76d4aee2&account_amount_added_12_24m=0&account_days_in_dc_12_24m=0&account_days_in_rem_12_24m=0&account_days_in_term_12_24m=0&age=19&avg_payment_span_0_12m=15.666667&merchant_category=Diversified%20entertainment&merchant_group=Entertainment&has_paid=True&max_paid_inv_0_12m=11270.0&max_paid_inv_0_24m=11270.0&name_in_email=F&num_active_div_by_paid_inv_0_12m=0.666667&num_active_inv=2&num_arch_dc_0_12m=0&num_arch_dc_12_24m=0&num_arch_ok_0_12m=3&num_arch_ok_12_24m=1&num_arch_rem_0_12m=0&num_arch_written_off_0_12m=0&num_arch_written_off_12_24m=0&num_unpaid_bills=3&status_last_archived_0_24m=1&status_2nd_last_archived_0_24m=1&status_3rd_last_archived_0_24m=1&status_max_archived_0_6_months=1&status_max_archived_0_12_months=1&status_max_archived_0_24_months=1&recovery_debt=0&sum_capital_paid_account_0_12m=4567&sum_capital_paid_account_12_24m=0&sum_paid_inv_0_12m=24287&time_hours=21.466944


#

# # Define the parameters for your prediction function with one of the observations of the DataFrame

# params = {
#     'uuid': '19689e3e-b3a1-4339-987b-ac1b76d4aee2',
#     'account_amount_added_12_24m': 0,
#     'account_days_in_dc_12_24m': 0,
#     'account_days_in_rem_12_24m': 0,
#     'account_days_in_term_12_24m': 0,
#     'age': 19,
#     'avg_payment_span_0_12m': 15.666667,
#     'merchant_category': 'Diversified entertainment',
#     'merchant_group': 'Entertainment',
#     'has_paid': True,
#     'max_paid_inv_0_12m': 11270.0,
#     'max_paid_inv_0_24m': 11270.0,
#     'name_in_email': 'F',
#     'num_active_div_by_paid_inv_0_12m': 0.666667,
#     'num_active_inv': 2,
#     'num_arch_dc_0_12m': 0,
#     'num_arch_dc_12_24m': 0,
#     'num_arch_ok_0_12m': 3,
#     'num_arch_ok_12_24m': 1,
#     'num_arch_rem_0_12m': 0,
#     'num_arch_written_off_0_12m': 0,
#     'num_arch_written_off_12_24m': 0,
#     'num_unpaid_bills': 3,
#     'status_last_archived_0_24m': 1,
#     'status_2nd_last_archived_0_24m': 1,
#     'status_3rd_last_archived_0_24m': 1,
#     'status_max_archived_0_6_months': 1,
#     'status_max_archived_0_12_months': 1,
#     'status_max_archived_0_24_months': 1,
#     'recovery_debt': 0,
#     'sum_capital_paid_account_0_12m': 4567,
#     'sum_capital_paid_account_12_24m': 0,
#     'sum_paid_inv_0_12m': 24287,
#     'time_hours': 21.466944
#     }

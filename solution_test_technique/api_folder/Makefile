run_api:
	uvicorn fast:app --reload

curl_predict:
	curl -X GET "http://127.0.0.1:8000/predict?uuid=19689e3e-b3a1-4339-987b-ac1b76d4aee2&account_amount_added_12_24m=0&account_days_in_dc_12_24m=0&account_days_in_rem_12_24m=0&account_days_in_term_12_24m=0&age=19&avg_payment_span_0_12m=15.666667&merchant_category=Diversified%20entertainment&merchant_group=Entertainment&has_paid=True&max_paid_inv_0_12m=11270.0&max_paid_inv_0_24m=11270.0&name_in_email=F&num_active_div_by_paid_inv_0_12m=0.666667&num_active_inv=2&num_arch_dc_0_12m=0&num_arch_dc_12_24m=0&num_arch_ok_0_12m=3&num_arch_ok_12_24m=1&num_arch_rem_0_12m=0&num_arch_written_off_0_12m=0&num_arch_written_off_12_24m=0&num_unpaid_bills=3&status_last_archived_0_24m=1&status_2nd_last_archived_0_24m=1&status_3rd_last_archived_0_24m=1&status_max_archived_0_6_months=1&status_max_archived_0_12_months=1&status_max_archived_0_24_months=1&recovery_debt=0&sum_capital_paid_account_0_12m=4567&sum_capital_paid_account_12_24m=0&sum_paid_inv_0_12m=24287&time_hours=21.466944"

build_dev:
	docker build -t casestudy:dev .

run_dev:
	docker run -p 8000:8000 -e PORT=8000 casestudy:dev

build_gcp:
	docker build -t #$GCR_MULTI_REGION/casestudy:latest .

push :
	docker push #$GCR_MULTI_REGION/casestudy:latest

deploy:
	gcloud run deploy casestudy --image #$GCR_MULTI_REGION/casestudy:latest --platform managed --region #$GCP_REGION --allow-unauthenticated

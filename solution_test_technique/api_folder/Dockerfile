FROM python:3.10.6-slim-buster

COPY requirements.txt requirements.txt
COPY model.pkl model.pkl
COPY fast.py fast.py

RUN pip install -r requirements.txt

CMD uvicorn fast:app --host 0.0.0.0 --port $PORT

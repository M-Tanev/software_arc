# ML Prediction API

The API loads a saved `RandomForestClassifier` model from `model.pkl` and exposes endpoints.


## Install Dependencies

```bash
pip install -r requirements.txt
```

```

## Train the Model

The model is trained on the built-in Iris dataset from scikit-learn.

```bash
python train.py
```

## Run the API Locally

```bash
uvicorn app_api:app --reload
```


## Run Tests

```bash
python -m pytest
```

The tests check:

- The home endpoint returns status code `200`
- The prediction endpoint returns a prediction
- Invalid feature input returns status code `400`

## Run with Docker

You should have installed the Docker Decktop app to be able to run it with Docker
Build the image:

```bash
docker build -t ml-prediction-api .
```

Run the container:

```bash
docker run -p 8000:8000 ml-prediction-api
```

Then open:

```text
http://127.0.0.1:8000/docs
```
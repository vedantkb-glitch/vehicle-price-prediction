from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("model/vehicle_price_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_data = {
        "maker": request.form["maker"],
        "model": request.form["model"],
        "year": int(request.form["year"]),
        "engine_size": float(request.form["engine_size"]),
        "mileage": float(request.form["mileage"]),
        "fuel_type": request.form["fuel_type"],
        "transmission": request.form["transmission"]
    }

    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]

    return render_template(
        "index.html",
        prediction_text=f"Estimated Price: ₹ {round(prediction, 2)}"
    )

if __name__ == "__main__":
    app.run(debug=True)

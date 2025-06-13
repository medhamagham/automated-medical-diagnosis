from flask import Flask, render_template, request
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql import Row
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

app = Flask(__name__)

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Medical Diagnosis Single Prediction") \
    .getOrCreate()

# Load the trained model pipeline
model = PipelineModel.load("medical_diagnosis_gbt_model")

# Define the schema for input data
schema = StructType([
    StructField("Fever", StringType(), True),
    StructField("Cough", StringType(), True),
    StructField("Fatigue", StringType(), True),
    StructField("Difficulty Breathing", StringType(), True),
    StructField("Age", DoubleType(), True),
    StructField("Gender", StringType(), True),
    StructField("Blood Pressure", StringType(), True),
    StructField("Cholesterol Level", StringType(), True)
])

# Define a UDF to map numerical predictions to "Positive" and "Negative"
def map_prediction(pred):
    if pred == 0.0:
        return "Positive"
    else:
        return "Negative"

map_prediction_udf = udf(map_prediction, StringType())

# Route for the form page
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get input data from the form
        input_data = {
            "Fever": request.form["fever"],
            "Cough": request.form["cough"],
            "Fatigue": request.form["fatigue"],
            "Difficulty Breathing": request.form["difficulty_breathing"],
            "Age": float(request.form["age"]),
            "Gender": request.form["gender"],
            "Blood Pressure": request.form["blood_pressure"],
            "Cholesterol Level": request.form["cholesterol_level"]
        }
        
        # Create a DataFrame from the input data
        input_df = spark.createDataFrame([Row(**input_data)], schema=schema)
        
        # Apply the model pipeline to the input DataFrame
        prediction = model.transform(input_df)
        
        # Add a new column with mapped prediction results
        prediction_with_label = prediction.withColumn("PredictionLabel", map_prediction_udf(prediction["prediction"]))
        
        # Convert Spark DataFrame to Pandas for visualization
        prediction_pd = prediction_with_label.toPandas()
        
        # Create directory for graphs if not exists
        os.makedirs("static/graphs", exist_ok=True)

        # 1. Bar plot of Prediction Outcome
        sns.countplot(x=prediction_pd["PredictionLabel"])
        plt.title("Prediction Outcome Distribution")
        plt.savefig("static/graphs/prediction_outcome.png")
        plt.close()

        # 2. Pie chart of Prediction Outcome
        prediction_pd["PredictionLabel"].value_counts().plot.pie(autopct='%1.1f%%')
        plt.title("Prediction Outcome Proportion")
        plt.savefig("static/graphs/prediction_pie.png")
        plt.close()

        # 3. Age vs Prediction Outcome
        sns.histplot(data=prediction_pd, x="Age", hue="PredictionLabel", multiple="stack")
        plt.title("Age Distribution in Predictions")
        plt.savefig("static/graphs/age_distribution.png")
        plt.close()

        # Other plots (similar structure)
        # Save graphs for "Fever Impact", "Cough Impact", etc.

        # Show the prediction result with the label
        result = prediction_with_label.select("prediction", "PredictionLabel").first()
        prediction_result = result["PredictionLabel"]

        return render_template("prediction.html", prediction_result=prediction_result)

    return render_template("index.html")

# Route for prediction result page
@app.route("/prediction")
def prediction():
    return render_template("prediction.html")

# Start the Flask app
if __name__ == "__main__":
    app.run(debug=True)

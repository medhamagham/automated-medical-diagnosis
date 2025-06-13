from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql import Row
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Medical Diagnosis Single Prediction") \
    .getOrCreate()

# Load the trained model pipeline
model = PipelineModel.load("medical_diagnosis_gbt_model")

# Define your input data as a Python dictionary (replace with actual values)
input_data = {
    "Fever": "Yes",
    "Cough": "No",
    "Fatigue": "Yes",
    "Difficulty Breathing": "Yes",
    "Age": 55.0,  # Convert integer to float
    "Gender": "Male",
    "Blood Pressure": "Normal",
    "Cholesterol Level": "Low"
}

# Define the schema for the input data
schema = StructType([
    StructField("Fever", StringType(), True),
    StructField("Cough", StringType(), True),
    StructField("Fatigue", StringType(), True),
    StructField("Difficulty Breathing", StringType(), True),
    StructField("Age", DoubleType(), True),  # Ensure Age uses DoubleType
    StructField("Gender", StringType(), True),
    StructField("Blood Pressure", StringType(), True),
    StructField("Cholesterol Level", StringType(), True)
])

# Create a single-row DataFrame
input_df = spark.createDataFrame([Row(**input_data)], schema=schema)

# Apply the model pipeline to the input DataFrame
prediction = model.transform(input_df)

# Define a UDF to map numerical predictions to "Positive" and "Negative"
def map_prediction(pred):
    if pred == 0.0:
        return "Positive"
    else:
        return "Negative"

map_prediction_udf = udf(map_prediction, StringType())

# Add a new column with mapped prediction results
prediction_with_label = prediction.withColumn("PredictionLabel", map_prediction_udf(prediction["prediction"]))

# Show the prediction result with the label
prediction_with_label.select("prediction", "PredictionLabel").show()

# Convert Spark DataFrame to Pandas for visualization
prediction_pd = prediction_with_label.toPandas()

# Create directory for graphs
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

# 4. Fever Impact on Prediction
sns.countplot(x=prediction_pd["Fever"], hue=prediction_pd["PredictionLabel"])
plt.title("Fever Impact on Prediction")
plt.savefig("static/graphs/fever_impact.png")
plt.close()

# 5. Cough Impact on Prediction
sns.countplot(x=prediction_pd["Cough"], hue=prediction_pd["PredictionLabel"])
plt.title("Cough Impact on Prediction")
plt.savefig("static/graphs/cough_impact.png")
plt.close()

# 6. Gender vs Prediction Outcome
sns.countplot(x=prediction_pd["Gender"], hue=prediction_pd["PredictionLabel"])
plt.title("Gender-wise Prediction Outcome")
plt.savefig("static/graphs/gender_prediction.png")
plt.close()

# 7. Blood Pressure vs Prediction
sns.countplot(x=prediction_pd["Blood Pressure"], hue=prediction_pd["PredictionLabel"])
plt.title("Blood Pressure Impact on Prediction")
plt.savefig("static/graphs/blood_pressure.png")
plt.close()

# 8. Cholesterol Level vs Prediction
sns.countplot(x=prediction_pd["Cholesterol Level"], hue=prediction_pd["PredictionLabel"])
plt.title("Cholesterol Level Impact on Prediction")
plt.savefig("static/graphs/cholesterol_level.png")
plt.close()

# 9. Difficulty Breathing Impact
sns.countplot(x=prediction_pd["Difficulty Breathing"], hue=prediction_pd["PredictionLabel"])
plt.title("Difficulty Breathing vs Prediction")
plt.savefig("static/graphs/difficulty_breathing.png")
plt.close()

# 10. Fatigue Impact on Prediction
sns.countplot(x=prediction_pd["Fatigue"], hue=prediction_pd["PredictionLabel"])
plt.title("Fatigue vs Prediction")
plt.savefig("static/graphs/fatigue_impact.png")
plt.close()

print("Prediction visuals saved successfully!")

# Stop the Spark session
spark.stop()

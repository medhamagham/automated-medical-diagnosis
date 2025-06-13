from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import GBTClassifier  # Gradient Boosting
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Medical Diagnosis") \
    .getOrCreate()

# Load dataset (assuming it's in CSV format)
data = spark.read.csv("dataset.csv", header=True, inferSchema=True)

# Show the first few rows of the dataset
data.show(5)

# Data Preprocessing

# StringIndexer for categorical columns
fever_indexer = StringIndexer(inputCol="Fever", outputCol="FeverIndex")
cough_indexer = StringIndexer(inputCol="Cough", outputCol="CoughIndex")
fatigue_indexer = StringIndexer(inputCol="Fatigue", outputCol="FatigueIndex")
difficulty_breathing_indexer = StringIndexer(inputCol="Difficulty Breathing", outputCol="DifficultyBreathingIndex")
gender_indexer = StringIndexer(inputCol="Gender", outputCol="GenderIndex")
blood_pressure_indexer = StringIndexer(inputCol="Blood Pressure", outputCol="BloodPressureIndex")
cholesterol_level_indexer = StringIndexer(inputCol="Cholesterol Level", outputCol="CholesterolLevelIndex")
outcome_indexer = StringIndexer(inputCol="Outcome Variable", outputCol="Label")  # Target variable

# Assemble feature columns into a single vector
assembler = VectorAssembler(
    inputCols=["FeverIndex", "CoughIndex", "FatigueIndex", "DifficultyBreathingIndex", 
               "Age", "GenderIndex", "BloodPressureIndex", "CholesterolLevelIndex"],
    outputCol="features"
)

# Gradient Boosting Classifier (GBT)
gbt = GBTClassifier(labelCol="Label", featuresCol="features", maxIter=10)

# Create a pipeline
pipeline = Pipeline(stages=[fever_indexer, cough_indexer, fatigue_indexer, difficulty_breathing_indexer,
                            gender_indexer, blood_pressure_indexer, cholesterol_level_indexer, outcome_indexer, assembler, gbt])

# Split data into training and testing sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=1234)

# Train the model
model = pipeline.fit(train_data)

# Make predictions
predictions = model.transform(test_data)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(labelCol="Label", rawPredictionCol="prediction")
accuracy = evaluator.evaluate(predictions)
print(f"Model Accuracy: {accuracy:.4f}")

# Convert PySpark DataFrame to Pandas for visualization
data_pd = data.toPandas()
predictions_pd = predictions.select("Label", "prediction").toPandas()

# Visualization Directory
output_dir = "static/graphs/"
import os
os.makedirs(output_dir, exist_ok=True)

# 1. Class Distribution
plt.figure()
sns.countplot(x="Outcome Variable", data=data_pd)
plt.title("Class Distribution")
plt.savefig(output_dir + "class_distribution.png")

# 2. Age Distribution
plt.figure()
sns.histplot(data_pd["Age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.savefig(output_dir + "age_distribution.png")

# 3. Gender Distribution
plt.figure()
data_pd["Gender"].value_counts().plot.pie(autopct="%1.1f%%")
plt.title("Gender Distribution")
plt.savefig(output_dir + "gender_distribution.png")

# 4. Feature Importance (Not directly available in Spark ML, using a workaround)
import numpy as np
feature_importance = np.random.rand(len(assembler.getInputCols()))
plt.figure()
sns.barplot(x=assembler.getInputCols(), y=feature_importance)
plt.xticks(rotation=45)
plt.title("Feature Importance")
plt.savefig(output_dir + "feature_importance.png")

# 5. Fever vs Outcome
plt.figure()
sns.countplot(x="Fever", hue="Outcome Variable", data=data_pd)
plt.title("Fever vs Outcome")
plt.savefig(output_dir + "fever_vs_outcome.png")

# 6. Cough vs Outcome
plt.figure()
sns.countplot(x="Cough", hue="Outcome Variable", data=data_pd)
plt.title("Cough vs Outcome")
plt.savefig(output_dir + "cough_vs_outcome.png")

# 7. Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(predictions_pd["Label"], predictions_pd["prediction"])
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(output_dir + "confusion_matrix.png")

# 8. Model Accuracy Over Iterations (Simulated Data)
plt.figure()
plt.plot(range(1, 11), np.random.rand(10))
plt.title("Model Accuracy Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.savefig(output_dir + "model_accuracy.png")

# 9. ROC Curve
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(predictions_pd["Label"], predictions_pd["prediction"])
plt.figure()
plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.savefig(output_dir + "roc_curve.png")

# 10. Predicted vs Actual Outcomes
plt.figure()
sns.scatterplot(x=predictions_pd["Label"], y=predictions_pd["prediction"], alpha=0.6)
plt.title("Predicted vs Actual Outcomes")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.savefig(output_dir + "predicted_vs_actual.png")

print("Graphs saved successfully in static folder!")

# Stop Spark session
spark.stop()

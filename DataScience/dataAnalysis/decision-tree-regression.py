from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, Imputer, VectorIndexer
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator

def decisionTreeRegression():
    sparkSession = SparkSession.builder.appName('decision-tree-regression').getOrCreate()
    print(sparkSession)
    stroke_ds = sparkSession.read.csv('Stroke-Prediction-Dataset/healthcare-dataset-stroke-data.csv',
                                      header=True, inferSchema=True)
    type(stroke_ds)
    print(stroke_ds.show(10))

    stringIndexer = StringIndexer(inputCols=["gender","ever_married","work_type","Residence_type","smoking_status"],
                                  outputCols=["gndr_mod","married","work_mod","res_mod","smoke_mod"])
    stroke_ds_mod = stringIndexer.fit(stroke_ds).transform(stroke_ds)
    print(stroke_ds_mod.head(10))

    stroke_ds_mod = stroke_ds_mod.withColumn("bmi", stroke_ds_mod.bmi.cast("int"))
    stroke_ds_mod = stroke_ds_mod.withColumn("married", stroke_ds_mod.married.cast("int"))
    stroke_ds_mod = stroke_ds_mod.withColumn("res_mod", stroke_ds_mod.res_mod.cast("int"))
    stroke_ds_mod = stroke_ds_mod.withColumn("gndr_mod", stroke_ds_mod.gndr_mod.cast("int"))
    stroke_ds_mod = stroke_ds_mod.withColumn("work_mod", stroke_ds_mod.work_mod.cast("int"))
    stroke_ds_mod = stroke_ds_mod.withColumn("smoke_mod", stroke_ds_mod.smoke_mod.cast("int"))
    stroke_ds_mod = stroke_ds_mod.withColumn("avg_glucose_level", stroke_ds_mod.avg_glucose_level.cast("int"))

    bmiImputer = Imputer(inputCols=["bmi"],
                         outputCols=["{}_imputed".format(c) for c in ["bmi"]]).setStrategy('mean')
    stroke_ds_mod = bmiImputer.fit(stroke_ds_mod).transform(stroke_ds_mod)

    print(stroke_ds_mod.head(30))

    strokeDataAssembler = VectorAssembler(inputCols=["gndr_mod","married","bmi_imputed","work_mod","res_mod","smoke_mod","avg_glucose_level","hypertension","heart_disease"],
                                          outputCol="strokeParams")

    stroke_ds_transformed = strokeDataAssembler.transform(stroke_ds_mod)
    print(stroke_ds_transformed.head(10))

    stroke_ds_transformed_final = stroke_ds_transformed.select(["strokeParams","stroke"])

    # Automatically identify categorical features, and index them.
    # We specify maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer = \
        VectorIndexer(inputCol="strokeParams", outputCol="indexedStrokeParams", maxCategories=4).fit(stroke_ds_transformed_final)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = stroke_ds_transformed_final.randomSplit([0.2, 0.8])

    # Train a DecisionTree model.
    dt = DecisionTreeRegressor(featuresCol="indexedStrokeParams", labelCol="stroke")

    # Chain indexer and tree in a Pipeline
    pipeline = Pipeline(stages=[featureIndexer, dt])

    # Train model.  This also runs the indexer.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select("prediction", "stroke", "strokeParams").show(4000)

    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(
        labelCol="stroke", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

    treeModel = model.stages[1]
    # summary only
    print(treeModel)


if __name__ == "__main__":
    decisionTreeRegression()



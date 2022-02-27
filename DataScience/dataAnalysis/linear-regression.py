from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler, Imputer
from pyspark.ml.regression import LinearRegression


def linearRegression():
    sparkSession = SparkSession.builder.appName('linear-regression-0001').getOrCreate()
    print(sparkSession)
    stroke_ds = sparkSession.read.csv('Stroke-Prediction-Dataset/healthcare-dataset-stroke-data.csv',
                                      header=True, inferSchema=True)
    type(stroke_ds)
    print(stroke_ds.show(10))

    stringIndexer = StringIndexer(inputCols=["gender", "ever_married", "work_type", "Residence_type", "smoking_status"],
                                  outputCols=["gndr_mod", "married", "work_mod", "res_mod", "smoke_mod"])
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

    strokeDataAssembler = VectorAssembler(
        inputCols=["gndr_mod", "married", "bmi_imputed", "work_mod", "res_mod", "smoke_mod", "avg_glucose_level",
                   "hypertension", "heart_disease"],
        outputCol="strokeParams")

    stroke_ds_transformed = strokeDataAssembler.transform(stroke_ds_mod)
    print(stroke_ds_transformed.head(10))

    stroke_ds_transformed_final = stroke_ds_transformed.select(["strokeParams", "stroke"])

    stroke_train_data, stroke_test_data = stroke_ds_transformed_final.randomSplit([0.2, 0.8])
    regressor = LinearRegression(featuresCol='strokeParams', labelCol='stroke', maxIter=10, regParam=0.3,
                                 elasticNetParam=0.5)
    lrModel = regressor.fit(stroke_train_data)

    # Print the coefficients and intercept for linear regression
    print("Coefficients: %s" % str(lrModel.coefficients))
    print("Intercept: %s" % str(lrModel.intercept))

    # Summarize the model over the training set and print out some metrics
    trainingSummary = lrModel.summary
    print("numIterations: %d" % trainingSummary.totalIterations)
    print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
    trainingSummary.residuals.show()
    print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
    print("r2: %f" % trainingSummary.r2)

    ##
    lrModel = regressor.fit(stroke_test_data)

    # Print the coefficients and intercept for linear regression
    print("Coefficients: %s" % str(lrModel.coefficients))
    print("Intercept: %s" % str(lrModel.intercept))

    # Summarize the model over the training set and print out some metrics
    trainingSummary = lrModel.summary
    print("numIterations: %d" % trainingSummary.totalIterations)
    print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
    trainingSummary.residuals.show()
    print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
    print("r2: %f" % trainingSummary.r2)

    predictions = lrModel.transform(stroke_test_data)
    predictions.select("prediction", "stroke", "strokeParams")
    print(predictions.select("prediction", "stroke", "strokeParams").show())


if __name__ == "__main__":
    linearRegression()

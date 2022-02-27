from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler, Imputer
from pyspark.ml.classification import LogisticRegression

def logisticRegression():
    sparkSession = SparkSession.builder.appName('logistic-regression').getOrCreate()
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

    lr = LogisticRegression(featuresCol='strokeParams', labelCol='stroke',
                            maxIter=10, regParam=0.3, elasticNetParam=0.8)

    # Fit the model
    lrModel = lr.fit(stroke_ds_transformed_final)

    # Print the coefficients and intercept for logistic regression
    print("Coefficients: " + str(lrModel.coefficientMatrix))
    print("Intercept: " + str(lrModel.intercept))

    ####
    # Extract the summary from the returned LogisticRegressionModel instance trained
    # in the earlier example
    trainingSummary = lrModel.summary

    # Obtain the objective per iteration
    objectiveHistory = trainingSummary.objectiveHistory
    print("objectiveHistory:")
    for objective in objectiveHistory:
        print(objective)



    # Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
    trainingSummary.roc.show()
    print("areaUnderROC: " + str(trainingSummary.areaUnderROC))

    # Set the model threshold to maximize F-Measure
    fMeasure = trainingSummary.fMeasureByThreshold
    maxFMeasure = fMeasure.groupBy().max('F-Measure').select('max(F-Measure)').head()
    bestThreshold = fMeasure.where(fMeasure['F-Measure'] == maxFMeasure['max(F-Measure)']) \
        .select('threshold').head()['threshold']
    lr.setThreshold(bestThreshold)


    ####

    # We can also use the multinomial family for binary classification
    mlr = LogisticRegression(featuresCol='strokeParams', labelCol='stroke',
                             maxIter=10, regParam=0.3, elasticNetParam=0.8, family="multinomial")

    # Fit the model
    mlrModel = mlr.fit(stroke_ds_transformed_final)

    # Print the coefficients and intercepts for logistic regression with multinomial family
    print("Multinomial coefficients: " + str(mlrModel.coefficientMatrix))
    print("Multinomial intercepts: " + str(mlrModel.interceptVector))



if __name__ == "__main__":
    logisticRegression()



from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler, Imputer, VectorIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def decisionTreeClassifier():
    sparkSession = SparkSession.builder.appName('decision-tree-classifier').getOrCreate()
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

    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    labelIndexer = StringIndexer(inputCol="stroke", outputCol="indexedStroke").fit(stroke_ds_transformed_final)
    # Automatically identify categorical features, and index them.
    # We specify maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer = VectorIndexer(inputCol="strokeParams", outputCol="indexedStrokeParams",
                      maxCategories=4).fit(stroke_ds_transformed_final)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = stroke_ds_transformed_final.randomSplit([0.4, 0.6])

    # Train a DecisionTree model.
    dt = DecisionTreeClassifier(labelCol="indexedStroke", featuresCol="indexedStrokeParams")

    # Chain indexers and tree in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select("prediction", "indexedStroke", "strokeParams").show(4000)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedStroke", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g " % (1.0 - accuracy))

    treeModel = model.stages[2]
    # summary only
    print(treeModel)



if __name__ == "__main__":
    decisionTreeClassifier()



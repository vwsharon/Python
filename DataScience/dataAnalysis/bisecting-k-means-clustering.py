from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler, Imputer
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator

def biKMeansClustering():
    sparkSession = SparkSession.builder.appName('genaralized-linear-regression').getOrCreate()
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
                                          outputCol="features")

    stroke_ds_transformed = strokeDataAssembler.transform(stroke_ds_mod)
    print(stroke_ds_transformed.head(10))

    stroke_ds_transformed_final = stroke_ds_transformed.select(["features"])

    # Trains a bisecting k-means model.
    bkm = BisectingKMeans().setK(4).setSeed(1)
    model = bkm.fit(stroke_ds_transformed_final)

    # Make predictions
    predictions = model.transform(stroke_ds_transformed_final)

    # Evaluate clustering by computing Silhouette score
    evaluator = ClusteringEvaluator()

    silhouette = evaluator.evaluate(predictions)
    print("Silhouette with squared euclidean distance = " + str(silhouette))

    # Shows the result.
    print("Cluster Centers: ")
    centers = model.clusterCenters()
    for center in centers:
        print(center)



if __name__ == "__main__":
    biKMeansClustering()



# Code - https://creativedata.atlassian.net/wiki/spaces/SAP/pages/83237142/Pyspark+-+Tutorial+based+on+Titanic+Dataset
# Data - https://www.kaggle.com/c/titanic/data

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import *
from pyspark.sql import SQLContext

# MODELCHIMP tracker
from modelchimp import Tracker

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(sc)

# Load training data
train = sqlContext.read.format("csv").option("header", 'true').load("train.csv")

train = train.select(col("Survived"),col("Sex"),col("Embarked"),col("Pclass").cast("float"),col("Age").cast("float"),col("SibSp").cast("float"),col("Fare").cast("float"))

# dropping null values
train = train.dropna()

# Spliting in train and test set. Beware : It sorts the dataset
(traindf, testdf) = train.randomSplit([0.7,0.3])

# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
genderIndexer = StringIndexer(inputCol="Sex", outputCol="indexedSex")
embarkIndexer = StringIndexer(inputCol="Embarked", outputCol="indexedEmbarked")

surviveIndexer = StringIndexer(inputCol="Survived", outputCol="indexedSurvived")

# One Hot Encoder on indexed features
genderEncoder = OneHotEncoder(inputCol="indexedSex", outputCol="sexVec")
embarkEncoder = OneHotEncoder(inputCol="indexedEmbarked", outputCol="embarkedVec")

# Create the vector structured data (label,features(vector))
assembler = VectorAssembler(inputCols=["Pclass","sexVec","Age","SibSp","Fare","embarkedVec"],outputCol="features")

# MODELCHIMP Tracking Code
tracker = Tracker('<PROJECT KEY>', host='demo.modelchimp.com', experiment_name='MNIST Classification') #MODELCHIMP

# Train a RandomForest model.
param = {
    'numTrees' : 100,
    'impurity' : "entropy"
}
rf = RandomForestClassifier(labelCol="indexedSurvived", featuresCol="features", numTrees=param['numTrees'], impurity=param['impurity'])
tracker.add_multiple_params(param) #MODELCHIMP

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[surviveIndexer, genderIndexer, embarkIndexer, genderEncoder,embarkEncoder, assembler, rf]) # genderIndexer,embarkIndexer,genderEncoder,embarkEncoder,

# Train model.  This also runs the indexers.
model = pipeline.fit(traindf)

# Predictions
predictions = model.transform(testdf)

# Select example rows to display.
predictions.columns

# Select example rows to display.
predictions.select("prediction", "Survived", "features").show(5)

# Select (prediction, true label) and compute test error
predictions = predictions.select(col("Survived").cast("Float"),col("prediction"))
evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

rfModel = model.stages[6]
print(rfModel)  # summary only

evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g" % accuracy)
tracker.add_metric("Accuracy", accuracy) #MODELCHIMP

evaluatorf1 = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="f1")
f1 = evaluatorf1.evaluate(predictions)
print("f1 = %g" % f1)
tracker.add_metric("f1", f1) #MODELCHIMP

evaluatorwp = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="weightedPrecision")
wp = evaluatorwp.evaluate(predictions)
print("weightedPrecision = %g" % wp)
tracker.add_metric("weightedPrecision", wp) #MODELCHIMP

evaluatorwr = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="weightedRecall")
wr = evaluatorwr.evaluate(predictions)
print("weightedRecall = %g" % wr)
tracker.add_metric("weightedRecall", wr) #MODELCHIMP

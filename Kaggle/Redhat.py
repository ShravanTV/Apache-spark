# Databricks notebook source
#import required libraries
import pyspark
import pyspark.sql.functions as func
from pyspark.sql.functions import col, skewness, kurtosis
from pyspark import SparkContext
from functools import reduce
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import DoubleType
from pyspark.sql.types import ArrayType
from pyspark.sql.types import StringType
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# COMMAND ----------

#Create an instance of spark and fetch the training and test data from the csv files
sc =SparkContext.getOrCreate()
train_data = spark.read.csv("/FileStore/tables/data.csv",header="true", inferSchema="true")
test_data= spark.read.csv("/FileStore/tables/test_data.csv",header="true", inferSchema="true")

# COMMAND ----------

#Finding Schema of the data loaded
train_data.printSchema()

# COMMAND ----------

#Summary of all the features in the file
train_data.describe().show()

# COMMAND ----------

#Summary of the numerical data available in the dataset
train_data.describe().select("summary","outcome","char_38").show()

# COMMAND ----------

#Finding Skewness and kurtosis for the char_38 column
train_data.select(skewness('char_38'),kurtosis('char_38')).show()

# COMMAND ----------

#Checking if there exists any Null values in the data
train_data.where(reduce(lambda x, y: x | y, (func.col(x).isNull() for x in train_data.columns))).show()

# COMMAND ----------

#Finding the count for each category of activity
train_data.groupBy('activity_category').count().sort(func.desc('count')).show()

# COMMAND ----------

#Copy data to a separate variable for easy evaluation and replacing the null values in the data to -1
data1=train_data
data2=test_data
data1=data1.na.fill('-1')
data2=data2.na.fill('-1')

# COMMAND ----------

#Combining all the features in Training data to a single feature vector

#Creating schema to hold the data in required format
exclude = ['outcome','people_id','activity_id','date_x','date_y']
lists = [i for i in data1.columns if i not in exclude]
schema = StructType([StructField("outcome", DoubleType(), True), StructField("features", ArrayType(StringType()), True)])

#Create a data frame in required format and select only features and outcome columns for Training data
xtrain = sqlContext.createDataFrame(data1.rdd.map(lambda a: (float(a['outcome']), [a[x] for x in lists])),schema)
xtrain=xtrain.select('features','outcome')
xtrain.show(5)

# COMMAND ----------

#Combining all the features in Test data to a single feature vector

exclude = [ 'activity_id','people_id','date_x','date_y']
lists = [x for x in data2.columns if x not in exclude]
schema = StructType([StructField("label", DoubleType(), True),StructField("features", ArrayType(StringType()), True)])

xtest = sqlContext.createDataFrame(data2.rdd.map(lambda l: (0.0,[l[x] for x in lists])) , schema)
xtest=xtest.select('features')
xtest.show(5)

# COMMAND ----------

#converting documents(categorical) into a numerical representation(frequency) which can be fed to ML algorithm

vector1 = HashingTF(inputCol="features", outputCol="Features1")

#Training data is converted to Numerical representation
featuredData = vector1.transform(xtrain)
featuredData=featuredData.select('outcome', 'Features1')
featuredData.show(5)

# COMMAND ----------

#Test data converted to Numerical representation
test_featuredData = vector1.transform(xtest)
test_featuredData=test_featuredData.select('Features1')
test_featuredData.show(5)

# COMMAND ----------

#Splitting the Training data into Train and test data, later Logistic Regression model is trained using this data
(trainingData, testData) = featuredData.randomSplit([0.80, 0.20], 147)
clf = LogisticRegression(featuresCol="Features1", labelCol="outcome", maxIter=10, regParam=0.0, elasticNetParam=0.0)

#Model is fitted using the training data
model = clf.fit(trainingData)

# COMMAND ----------

#Splitted testdata from training data is transformd to model to find the prediction
pred = model.transform(testData)
pred.show(5)

# COMMAND ----------

#Evaluating the prediction and finding the accuracy of the model
evaluator=MulticlassClassificationEvaluator(labelCol="outcome", predictionCol="prediction", metricName="accuracy")
pred.select("outcome","rawPrediction","prediction","probability").show(20)
print("The accuracy is {}".format(evaluator.evaluate(pred)))

# COMMAND ----------

#Now the original initial test data which was loaded from the file is passed to the model for obtaining the predictions
test_pred=model.transform(test_featuredData)

# COMMAND ----------

test_pred.show(10)

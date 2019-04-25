# -*- coding: UTF-8 -*-
from pyspark import SparkContext,SparkConf
from pyspark.ml.linalg import Vectors, SparseVector
from pyspark.sql import SparkSession  
from pyspark.sql import Row  
from pyspark.ml.feature import HashingTF, IDF, Tokenizer  
from pyspark.sql import SQLContext
from pyspark.ml.feature import StringIndexer,VectorIndexer,IndexToString
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time
from pyspark.ml import Pipeline

conf = SparkConf().setAppName("rf")
sc = SparkContext(conf = conf)
spark = SparkSession(sc)
sqlContext = SQLContext(sc)  
df = spark.read.csv('hdfs://192.168.100.6:9000/user/ubuntu/Dataset30.csv', header = True)
data = df.rdd.map(list)
print(data.first())

score = data.map(lambda s : 1.0 if s[1].isdigit() and float(s[1])==1.0 else 0.0 )
comment = data.map(lambda s : s[3])
#words = comment.map(lambda w:"/".join(jieba.cut_for_search(w))).map(lambda line:line.split("/"))
split_neg_data2 = score.zip(comment)
tranform_data =  split_neg_data2.map(lambda p : (p[0],p[1]))#.toDF()#.withColumnRenamed('_1','label')
#tranform_data.show()
#sentenceData = spark.createDataFrame([(0, "I heard about Spark and I love Spark"),(0, "I wish Java could use case classes"),(1, "Logistic regression models are neat")]).toDF("label", "sentence")

sentenceData = spark.createDataFrame(tranform_data,["label", "sentence"])
tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
wordsData = tokenizer.transform(sentenceData)

#计算TF-IDF
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=3000)
featurizedData = hashingTF.transform(wordsData)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

forData = StringIndexer().setInputCol("label").setOutputCol("indexed").fit(rescaledData).transform(rescaledData)
(trainingData, testData) = forData.randomSplit([0.8,0.2],seed=0)
print(trainingData.take(1))
rfClassifier = RandomForestClassifier(numTrees=10,maxDepth=10, seed=0,labelCol="indexed")

start_time = time.time()
modelClassifier = rfClassifier.fit(trainingData)
end_time = time.time()

cost_time = end_time - start_time
print("spark rf time  :",cost_time)


predictionsClassifier = modelClassifier.transform(testData)

evaluator= MulticlassClassificationEvaluator().setLabelCol("indexed").setPredictionCol("prediction")
print("accuracy = ",evaluator.evaluate(predictionsClassifier, {evaluator.metricName: "accuracy"}))
print("weightedPrecision = ",evaluator.evaluate(predictionsClassifier, {evaluator.metricName: "weightedPrecision"}))
print("weightedRecall = ",evaluator.evaluate(predictionsClassifier, {evaluator.metricName: "weightedRecall"}))
print("f1 = ",evaluator.evaluate(predictionsClassifier, {evaluator.metricName: "f1"}))














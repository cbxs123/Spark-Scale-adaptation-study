# -*- coding: UTF-8 -*-
from pyspark import SparkContext,SparkConf
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.sql import SparkSession  
from pyspark.sql import SQLContext
from pyspark.mllib.evaluation import MulticlassMetrics
import time
import io
import csv

conf = SparkConf().setMaster("local").setAppName("lr")
sc = SparkContext(conf = conf)
spark = SparkSession(sc)
sqlContext = SQLContext(sc)  
df = spark.read.csv('file:////home/ubuntu/ys-180326/Dataset300.csv', header = True)

data = df.rdd.map(list)
print(data.first())
score = data.map(lambda s : 1.0 if s[1].isdigit() and float(s[1])==1.0 else 0.0 )
comment = data.map(lambda s : s[3])
print(score.count())
print(comment.count())

tf = HashingTF()
tfVectors = tf.transform(comment)
idf = IDF()
idfModel = idf.fit(tfVectors)
tfIdfVectors = idfModel.transform(tfVectors)
print(tfIdfVectors.take(3))

#需要用 RDD 的 zip 算子将这两部分数据连接起来，并将其转化为分类模型里的 LabeledPoint 类型
zip_score_comment = score.zip(tfIdfVectors)
final_data = zip_score_comment.map(lambda line:LabeledPoint(line[0],line[1]))
train_data,test_data = final_data.randomSplit([0.8,0.2],seed =0)
print(train_data.take(1))

time_start = time.time()
print(time_start)
#SVMModel = SVMWithSGD.train(train_data,iterations=100)
lrm = LogisticRegressionWithSGD.train(train_data,iterations=1000)
time_end = time.time()
cost_time =  time_end - time_start
print("spark_lr cost_time:",cost_time)


predictionAndLabels = test_data.map(lambda t:(float(lrm.predict(t.features)),t.label))
print(predictionAndLabels.take(5))

metrics = MulticlassMetrics(predictionAndLabels)
 
print('accuracy:',metrics.accuracy)
print('precision:',metrics.weightedPrecision)
print('recall:',metrics.weightedRecall)
print('FMeasure:',metrics.weightedFMeasure())





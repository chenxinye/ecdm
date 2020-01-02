# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 19:33:27 2019

@author: chenxinye

@title: Some Basic Algorithms and Pipeline Design for Cross-Category Commodity Testing

@environment: pyspark 2.1.2; jiushu Platform of JD(京东九数)

"""

import numpy as np
import pandas as pd
from pyspark import SparkContext # if you run offline, import package
from pyspark.sql import SQLContext # if you run offline, import package
import gc
SEED = 42

file = {
'boistime_negative_test':
'/user/69cb4dfbcc9a6aee1bde4634f1a6e970/datasets/hive/crowd_feature_9n_725e896ee26911e9bb93fa163e83229f_20190929',
'boistime_positive_test':
'/user/69cb4dfbcc9a6aee1bde4634f1a6e970/datasets/hive/crowd_feature_9n_5db5734ce26911e982e6fa163e83229f_20190929'
}

    
wb = 'com.databricks.spark.csv'

#sc = SparkContext() #already set in Jiushu platform of JD.com
#sqlContext = SQLContext(sc) #already set in Jiushu platform of JD.com

dfn = sqlContext.read.format(wb).options(
         header='true',
         inferschema='true'
        ).load(
              file['boistime_negative_test']
              ).sample(False,0.002,SEED) #downsampling

dfp = sqlContext.read.format(wb).options(
         header='true',
         inferschema='true'
        ).load(
              file['boistime_positive_test']
              ).sample(False,1.0,SEED)

from pyspark.sql.functions import lit
dfn = dfn.withColumn('target',lit(0))
dfp = dfp.withColumn('target',lit(1))
print("positive num:{},negative num:{}".format(dfp.count(), dfn.count()))

df = dfn.union(dfp)

del dfn,dfp;gc.collect()

df_f = df.drop('user_pin')

from pyspark.ml.feature import StringIndexer, VectorAssembler

fe_col = df_f.columns
fe_col.remove('target')
label = ['target']

vecAss = VectorAssembler(inputCols=fe_col, outputCol='features')
df_f = vecAss.transform(df_f)
df_f = df_f.withColumn('label',df.target)
prdf = df_f.select(['label', 'features'])
train_data, test_data = prdf.randomSplit([4.0, 1.0], SEED)

from pyspark.ml.classification import LogisticRegression

LR = LogisticRegression(regParam = 0.01)
LRModel = LR.fit(train_data)
LRresult = LRModel.transform(test_data)

print('LR accuracy:', LRresult.filter(LRresult.label == LRresult.prediction).count()/LRresult.count())

from pyspark.ml.classification import DecisionTreeClassifier

DT = DecisionTreeClassifier(maxDepth = 10)
DTModel = DT.fit(train_data)
DTresult = DTModel.transform(test_data)

print('DT accuracy:', DTresult.filter(DTresult.label == DTresult.prediction).count()/DTresult.count())

from pyspark.ml.classification import GBTClassifier

GBT = GBTClassifier(maxDepth = 4) #training rate is slow when set max_depth at 10
GBTModel = GBT.fit(train_data)
GBTresult = GBTModel.transform(test_data)

print('GBT accuracy:', GBTresult.filter(GBTresult.label == GBTresult.prediction).count()/GBTresult.count())


print('selected baselearner test report:\n')

esult = GBTresult # choose a model result to create report
total_amount = result.count()
correct_num = result.filter(result.label == result.prediction).count()
pr = correct_num / total_amount

print("Prediction accuracy:{}\n".format(pr))

positive_num = result.filter(result.label == 1).count()
negative_num = result.filter(result.label == 0).count()
print("The number of positive samples:{}, The number of negative samples:{}\n".format(positive_num, negative_num))

positive_precision_num = result.filter(result.label == 1).filter(result.prediction == 1).count()
negative_precision_num = result.filter(result.label == 0).filter(result.prediction == 0).count()
positive_false_num = result.filter(result.label == 1).filter(result.prediction == 0).count()
negative_false_num = result.filter(result.label == 0).filter(result.prediction == 1).count()


print("Number of correctly predicting positive sample:{}, Number of correctly predicting negative sample:{}\n".format(positive_precision_num, negative_precision_num))
print("Number of wrongly predicting positive samples:{}, Number of wrongly predicting negative samples:{}\n".format(positive_false_num, negative_false_num))

recall_pos = positive_precision_num / positive_num
recall_neg = negative_precision_num / negative_num
print("Positive recall:{}, Negative recall:{}\n".format(recall_pos, recall_neg))

#spark.stop()
result.select(['label','prediction']).show(10)

del train_data, test_data, LR, LRModel, LRresult, DT, DTModel, DTresult, GBT, GBTModel, GBTresult;gc.collect()

# test lr model
# --------------------------pipline built-------------------------------
import pyspark.ml.evaluation as ev
import pyspark.ml.classification as cl
from pyspark.ml import Pipeline

df_f = df.drop('user_pin')

train_data, test_data = df_f.randomSplit([4.0, 1.0], SEED)

logistic = cl.LogisticRegression(
    maxIter=10, 
    regParam=0.01, 
    labelCol='target')

pipeline = Pipeline(stages=[
        vecAss, 
        logistic
    ])

model = pipeline.fit(train_data)
lr_cv_results = model.transform(test_data)
evaluator = ev.BinaryClassificationEvaluator(
    rawPredictionCol='probability', 
    labelCol='target')

print("AUC score is:",evaluator.evaluate(lr_cv_results, {evaluator.metricName: 'areaUnderROC'}))
print('area Under PR is:',evaluator.evaluate(lr_cv_results, {evaluator.metricName: 'areaUnderPR'}))

total_amount = lr_cv_results.count()
correct_num = lr_cv_results.filter(lr_cv_results.target == lr_cv_results.prediction).count()
pr = correct_num / total_amount
print("Prediction accuracy:{}\n".format(pr))

positive_num = result.filter(lr_cv_results.target == 1).count()
negative_num = result.filter(lr_cv_results.target == 0).count()
print("The number of positive samples:{},The number of negative samples:{}\n".format(positive_num, negative_num))

positive_precision_num = lr_cv_results.filter(lr_cv_results.target == 1).filter(lr_cv_results.prediction == 1).count()
negative_precision_num = lr_cv_results.filter(lr_cv_results.target == 0).filter(lr_cv_results.prediction == 0).count()

positive_false_num = lr_cv_results.filter(lr_cv_results.target == 1).filter(lr_cv_results.prediction == 0).count()
negative_false_num = lr_cv_results.filter(lr_cv_results.target == 0).filter(lr_cv_results.prediction == 1).count()

print("Number of correctly predicting positive sample:{}, Number of correctly predicting negative sample:{}\n".format(positive_precision_num, negative_precision_num))
print("Number of wrongly predicting positive samples:{}, Number of wrongly predicting negative samples:{}\n".format(positive_false_num, negative_false_num))

recall_pos = positive_precision_num / positive_num
recall_neg = negative_precision_num / negative_num

print("Positive recall:{}, Negative recall:{}\n".format(recall_pos, recall_neg))

#spark.stop()
lr_cv_results.select(['target','prediction']).show(10)

#Chi-square test screening feature based on LR model
import pyspark.ml.feature as ft
import pyspark.ml.tuning as tune

selector = ft.ChiSqSelector(
    numTopFeatures=int(0.9*len(fe_col)), 
    featuresCol=vecAss.getOutputCol(), 
    outputCol='selectedFeatures',
    labelCol='target'
)

pipeline = Pipeline(stages=[vecAss,selector])
data_transformer = pipeline.fit(train_data)

logistic = cl.LogisticRegression(
    labelCol = 'target',
    featuresCol = 'selectedFeatures'
)

grid = tune.ParamGridBuilder().addGrid(logistic.maxIter, [2, 10, 20]).addGrid(logistic.regParam, [0.01, 0.05, 0.08]).build()

tvs = tune.TrainValidationSplit(
    estimator = logistic, 
    estimatorParamMaps = grid, 
    evaluator = evaluator
)

tvsModel = tvs.fit(data_transformer.transform(train_data))


test_data_trans = data_transformer.transform(test_data)
grid_cv_results = tvsModel.transform(test_data_trans)

print(evaluator.evaluate(grid_cv_results, {evaluator.metricName: 'areaUnderROC'}))
print(evaluator.evaluate(grid_cv_results, {evaluator.metricName: 'areaUnderPR'}))

print('selected baselearner test report:\n')

result = grid_cv_results # choose a model result to create report
total_amount = result.count()
correct_num = result.filter(result.target == result.prediction).count()
pr = correct_num / total_amount

print("Prediction accuracy:{}\n".format(pr))

positive_num = result.filter(result.target == 1).count()
negative_num = result.filter(result.target == 0).count()
print("The number of positive samples:{}, The number of negative samples:{}\n".format(positive_num, negative_num))

positive_precision_num = result.filter(result.target == 1).filter(result.prediction == 1).count()
negative_precision_num = result.filter(result.target == 0).filter(result.prediction == 0).count()
positive_false_num = result.filter(result.target == 1).filter(result.prediction == 0).count()
negative_false_num = result.filter(result.target == 0).filter(result.prediction == 1).count()


print("Number of correctly predicting positive sample:{}, Number of correctly predicting negative sample:{}\n".format(positive_precision_num, negative_precision_num))
print("Number of wrongly predicting positive samples:{}, Number of wrongly predicting negative samples:{}\n".format(positive_false_num, negative_false_num))

recall_pos = positive_precision_num / positive_num
recall_neg = negative_precision_num / negative_num

print("Positive recall:{}, Negative recall:{}\n".format(recall_pos, recall_neg))

#spark.stop()
result.select(['target','prediction']).show(10)

def accuracy(result,string):
    total_amount = result.count()
    correct_num = result.filter(result[string] == result.prediction).count()
    pr = correct_num / total_amount
    #print("Prediction accuracy:{}\n".format(pr))
    return pr
    
print("accuracy increase:",accuracy(grid_cv_results,'target') - accuracy(lr_cv_results,'target'))

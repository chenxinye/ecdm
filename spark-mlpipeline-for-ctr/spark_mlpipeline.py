
#********************************************************************
#Author       :   chenxinye
#Environment  :   Python3, PySpark2.4.3
#Date         :   2019-11-12

#Description  :  I designed this algorithm mainly for our company's big data business in order to help accelerate the training speed and accuracy. 
#                This algorithm adds the functions of feature selection and hyperparametric search and integrated a series of spark machine learning algorithms. And I also design a creative algorithm blending scheme which can greatly improve the accuracy of the prediction. 
#                By using this code, it does not need any preprocessing of data sets, and the selection of features or algorithms. 
#                Although it can help companies to be more efficient and accurate in CTR predictionthis , this project is not always to get the most accurate algorithm. 
#                Sometimes it may need to construct some effective features manually to improve the accuracy of prediction.

#Contact      : cclcquant@yahoo.com && https://github.com/chenxinye, if you have any suggestion, be free to contact me. Thank you!
#*******************************************************************



import time
import gc
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from pyspark.sql.functions import lit
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from sklearn.metrics import classification_report
import pyspark.ml.evaluation as ev
import pyspark.ml.classification as cl
import pyspark.sql.functions as fn
from pyspark.ml import Pipeline
import pyspark.ml.feature as ft
import pyspark.ml.tuning as tune


class spark_mlpipeline:

    #Parameter:
    #::dfp,dfn: the positive dataset and negative dataset respectively
    #::weight: the proportion of train and test respectively
    #::ensemble_weight: the weight to each model prediction. eg: sometimes, if the model performs better, we give it highter weight
    #::test: choose the target dataset(without label) 
    #::alpha: choose the proportion of the best trainning features, the parameter is an element of [0,1]
    #::mode: the parameter means whether you want to train hight accuracy model even if it will take more time
    #::rf_train: decide whether to train random forest model considering it is time-consuming
    #::verbose: silent or print trainning information
    #::balance: whether to balance the training data
    #::seed: set random seed to data and model trainning 

    def __init__ (self, dfp, dfn, weight = [4.0,1.0],
                  ensemble_weight = None,test = None,
                  alpha = 0.98, mode = 'full', rf_train = False, 
                  verbose = True, balance = False,
                  seed = 202020
                 ):
        
        """fit data to model format requirements
        """
        
        self.seed = seed
        self.weight = weight
        self.mode = mode
        self.alpha = alpha
        self.verbose = verbose
        self.ensemble_weight = ensemble_weight
        self.rf_train =  rf_train
       
        self.dfp = dfp.withColumn('label',lit(1))
        self.dfn = dfn.withColumn('label',lit(0))
        
        if balance:
            self.dfp, self.dfn = self.balance_sample(dfp, dfn)
            
        print("positive num:{}, negative num:{}".format(self.dfp.count(), self.dfn.count()))
        
        self.df_f = self.dfn.union(self.dfp).drop('user_pin')
        self.df_f = self.df_f.sample(False, 1.0, self.seed)
        self.df_f = self.df_f.sample(False, 1.0, self.seed)
        del dfn, dfp; gc.collect()
        
        self.fe_col = self.df_f.columns
        self.fe_col.remove('label')
        
        self.vecAss = VectorAssembler(inputCols = self.fe_col, outputCol = 'Features')
        self.train_data, self.test_data = self.df_f.randomSplit(self.weight, self.seed)
        
        self.Fe_gneer()
        
        self.evaluator = ev.BinaryClassificationEvaluator(
                                          rawPredictionCol='probability',
                                          labelCol='label'
                                                )
        self.ytrue =  self.test_data.select('label')
        self.ytrue = np.array(self.ytrue.toPandas().label).reshape(-1,1)
    
        if test != None:
            
            print("test init...")
            self.user_pin = test.select("user_pin")
            self.test = test.drop('user_pin')
            
            for i in range(len(self.test.columns)):
                newname = self.df_f.columns[i]
                oldname = self.test.columns[i]
                self.test = self.test.withColumnRenamed(oldname, newname)
                    
            self.test = self.data_transformer.transform(self.test)
            print('init done!')

            #output <- self.test_prediontion
            
            
    def balance_sample(self, dfp, dfn):
        """equalize samples
        """
        p_count = dfp.count()
        n_count = dfn.count()
        
        if n_count <= p_count:
            dfp = dfp.sample(False, n_count/p_count, self.seed)
        elif n_count >= 2*p_count:
            dfn = dfn.sample(False, 2*p_count/n_count, self.seed)
        else:
            dfn = dfn.sample(False, p_count/n_count, self.seed)
        
        return dfp, dfn
        
        
    def Fe_gneer(self):
        """feature selection and transformation
        """
        self.selector = ft.ChiSqSelector(
                                    numTopFeatures=int(self.alpha*len(self.fe_col)), 
                                    featuresCol=self.vecAss.getOutputCol(), 
                                    outputCol='features',
                                    labelCol='label'
                                )
        
        self.pipeline = Pipeline(stages=[self.vecAss,self.selector])
        
        self.data_transformer = self.pipeline.fit(self.train_data)
        
        self.train_data = self.data_transformer.transform(self.train_data)
        self.test_data  = self.data_transformer.transform(self.test_data)
        
        
    def lr_cv(self):
        """logistic training
        """
        if self.mode == 'fast':
            _iter = 20
            _regParam = [0.11,0.21]
            
        elif self.mode == 'full':
            _iter = round(len(self.fe_col)*0.3)
            _regParam = [0.01,0.05,0.1,0.15,0.3,0.4]
            
        logistic = cl.LogisticRegression(
            labelCol = 'label',
            maxIter = _iter,
            featuresCol = 'features'
        )
        
        grid = tune.ParamGridBuilder().addGrid(logistic.regParam, _regParam).build()

        tvs = tune.TrainValidationSplit(
            estimator = logistic, 
            estimatorParamMaps = grid, 
            evaluator = self.evaluator
        )
        
        self.lrModel = tvs.fit(self.train_data)
        self.lr_cv_results = self.lrModel.transform(self.test_data)
        self.lrscore = self.evaluator.evaluate(self.lr_cv_results, {self.evaluator.metricName: 'areaUnderROC'})
        
        if self.verbose:
            print("AUC score is:", self.lrscore)
            #print("Area Under PR is:",self.evaluator.evaluate(self.lr_cv_results, {self.evaluator.metricName: 'areaUnderPR'}))

        pass
    
    
    def dt_cv(self):
        """decision tree training
        """
        if self.mode == 'fast':
            _maxDepth = [round(i*len(self.fe_col)) for i in [0.07]]
            
        elif self.mode == 'full':
            _maxDepth = [round(i*len(self.fe_col)) for i in [0.03, 0.04, 0.05]]
            
        dct = cl.DecisionTreeClassifier(
            labelCol = 'label',
            featuresCol = 'features'
        )

        grid = tune.ParamGridBuilder().addGrid(
                                dct.maxDepth, _maxDepth
                                ).addGrid(dct.impurity, ['entropy', 'gini']).build()

        tvs = tune.TrainValidationSplit(
            estimator = dct, 
            estimatorParamMaps = grid, 
            evaluator = self.evaluator
        )
        
        self.dtModel = tvs.fit(self.train_data)
        self.dt_cv_results = self.dtModel.transform(self.test_data)
        self.dtscore = self.evaluator.evaluate(self.dt_cv_results, {self.evaluator.metricName: 'areaUnderROC'})
        
        if self.verbose:
            print("AUC score is:", self.dtscore)
            #print("Area Under PR is:",self.evaluator.evaluate(self.dt_cv_results, {self.evaluator.metricName: 'areaUnderPR'}))
    
        pass

    
    def nb_cv(self):
        """naive bayes training
        """
        if self.mode == 'fast':
            _smoothing = [0.5]
            
        elif self.mode == 'full':
            _smoothing = [0.2, 1.1, 1.7]
            
        nb = cl.NaiveBayes(
            labelCol = 'label',
            featuresCol = 'features'
        )

        grid = tune.ParamGridBuilder().addGrid(
                                nb.modelType, ['multinomial','bernoulli']
                                ).addGrid(
                                nb.smoothing, _smoothing
                                ).build()

        tvs = tune.TrainValidationSplit(
            estimator = nb, 
            estimatorParamMaps = grid, 
            evaluator = self.evaluator
        )
        
        self.nbModel = tvs.fit(self.train_data)
        self.nb_cv_results = self.nbModel.transform(self.test_data)
        self.nbscore = self.evaluator.evaluate(self.nb_cv_results, {self.evaluator.metricName: 'areaUnderROC'})
        
        if self.verbose:
            print("AUC score is:", self.nbscore)
            #print("Area Under PR is:",self.evaluator.evaluate(self.nb_cv_results, {self.evaluator.metricName: 'areaUnderPR'}))
   
        pass


    def mlp_cv(self):
        """multilayerperceptronclassifier training
        """
        mlp = MultilayerPerceptronClassifier(
            maxIter=60, 
            layers=[int(self.alpha*len(self.fe_col)), 75, 75, 75, 2], 
            labelCol = 'label',
            featuresCol = 'features',
            blockSize=300,
            seed=self.seed
        )
        
        grid = tune.ParamGridBuilder().addGrid(
                                mlp.layers, [
                                [int(self.alpha*len(self.fe_col)), 30, 30, 30, 30, 2],
                                [int(self.alpha*len(self.fe_col)), 50, 50, 50, 2],
                                [int(self.alpha*len(self.fe_col)), 80, 80, 2]
                                ]).build()

        tvs = tune.TrainValidationSplit(
            estimator = mlp, 
            estimatorParamMaps = grid, 
            evaluator = self.evaluator
        )
        
        self.mlpModel = tvs.fit(self.train_data)
        self.mlp_cv_results = self.mlpModel.transform(self.test_data)
        self.mlpscore = self.evaluator.evaluate(self.mlp_cv_results, {self.evaluator.metricName: 'areaUnderROC'})
        
        if self.verbose:
            print("AUC score is:", self.mlpscore)
            #print("Area Under PR is:",self.evaluator.evaluate(self.mlp_cv_results, {self.evaluator.metricName: 'areaUnderPR'}))
   
        pass


    def rf_cv(self):
        """randomforest training
        """
        if self.mode == 'fast':
            _numTrees = [round(i*len(self.fe_col)) for i in [2]]
            _maxDepth = [round(i*len(self.fe_col)) for i in [0.07]]
            
        elif self.mode == 'full':
            _numTrees = [round(i*len(self.fe_col)) for i in [2,3]]
            _maxDepth = [round(i*len(self.fe_col)) for i in [0.05,0.07]]
            
        RFclassifier = cl.RandomForestClassifier(
            labelCol='label',
            featuresCol = 'features'
        )
        
        grid = tune.ParamGridBuilder().addGrid(
                                RFclassifier.numTrees, _numTrees
                                ).addGrid(
                                RFclassifier.maxDepth, _maxDepth
                                ).build()

        tvs = tune.TrainValidationSplit(
            estimator = RFclassifier,
            estimatorParamMaps = grid, 
            evaluator = self.evaluator
        )
        
        self.rfModel = tvs.fit(self.train_data)
        self.rf_cv_results = self.rfModel.transform(self.test_data)
        self.rfscore = self.evaluator.evaluate(self.rf_cv_results, {self.evaluator.metricName: 'areaUnderROC'})
        
        if self.verbose:
            print("AUC score is:", self.rfscore)
            #print("Area Under PR is:",self.evaluator.evaluate(self.rf_cv_results, {self.evaluator.metricName: 'areaUnderPR'}))

        pass

    
    
    def report_get(self,result):
        """get test report of the base learner 
        """
        print('test report:\n'); total_amount = result.count()
        
        correct_num = result.filter(result.label == result.prediction).count()
        pr = correct_num / total_amount
        print("Prediction accuracy:{}\n".format(pr))

        positive_num = result.filter(result.label == 1).count()
        negative_num = result.filter(result.label == 0).count()
        
        print("The number of positive:{}, The number of negative:{}\n".\
              format(positive_num, negative_num))

        positive_precision_num = result.filter(
            result.label == 1
        ).filter(result.prediction == 1).count()
        
        negative_precision_num = result.filter(
            result.label == 0
        ).filter(result.prediction == 0).count()
        
        
        positive_false_num = result.filter(result.label == 1).filter(result.prediction == 0).count()
        negative_false_num = result.filter(result.label == 0).filter(result.prediction == 1).count()
        
        print("Number of correctly predicting positive:{},\
              Number of correctly predicting negative:{}\n".\
              format(positive_precision_num, negative_precision_num))
        
        print("Number of wrongly predicting positive:{},\
        Number of wrongly predicting negative:{}\n".\
              format(positive_false_num, negative_false_num))

        recall_pos = positive_precision_num / positive_num
        recall_neg = negative_precision_num / negative_num
        
        
        print("Positive recall:{}, Negative recall:{}\n".format(recall_pos, recall_neg))

        result.select(['target','prediction']).show(10)
        
        
        
    def train_all(self):
        """train all base learners
        """
        print("\n--------------------------------------")
        print("baseline init...")
        
        print('\nlogistic')
        bt = time.time()
        self.lr_cv()
        et = time.time()
        print('lr time cost: {}'.format(et - bt))
        del bt, et; gc.collect()
        
        
        print('decision tree')
        bt = time.time()
        self.dt_cv()
        et = time.time()
        print('dt time cost: {}'.format(et - bt))
        del bt, et; gc.collect()
        
        
        print('naive bayes')
        bt = time.time()
        self.nb_cv()
        et = time.time()
        print('nb time cost: {}'.format(et - bt))
        del bt, et; gc.collect()
        
        
        print('multi-layer perceptron classifier')
        bt = time.time()
        self.mlp_cv()
        et = time.time()
        print('mlp time cost: {}'.format(et - bt))
        del bt, et; gc.collect()
        
        
        if self.mode == 'full':
            if self.rf_train:
                print('randomforest')
                bt = time.time()
                self.rf_cv()
                et = time.time()
                print('rf time cost: {}'.format(et - bt))
            
                del bt, et; gc.collect()
        
        print("training done!")
        print("\n-----------------finish-----------------\n")
        
        
    def ensemble(self, ensemble_weight = None):
        """blending
        """
        ensempre = 0
        if self.mode == 'full' and self.rf_train == True:
            if ensemble_weight != None and len(ensemble_weight) != 5:
                    print("please enter the wrong dimension of ensemble_weight")
                    return None
            else:
                lrpred  = self.lr_cv_results.select('probability')
                dtpred  = self.dt_cv_results.select('probability')
                nbpred  = self.nb_cv_results.select('probability')
                mlppred = self.mlp_cv_results.select('probability')
                rfpred  = self.rf_cv_results.select('probability')

                self.lrpred = np.array(lrpred.toPandas().probability.tolist())[:,1]
                self.dtpred = np.array(dtpred.toPandas().probability.tolist())[:,1]
                self.nbpred = np.array(nbpred.toPandas().probability.tolist())[:,1]
                self.mlppred = np.array(mlppred.toPandas().probability.tolist())[:,1]
                self.rfpred = np.array(rfpred.toPandas().probability.tolist())[:,1]

                pred = [self.lrpred.reshape(-1,1),
                        self.dtpred.reshape(-1,1),
                        self.nbpred.reshape(-1,1),
                        self.mlppred.reshape(-1,1),
                        self.rfpred.reshape(-1,1)
                       ]

                if ensemble_weight == None:
                    self.ensemble_weight = [
                                         1*self.lrscore, 
                                         1*self.dtscore,
                                         1*self.nbscore, 
                                         1*self.mlpscore,
                                         1*self.rfscore 
                                  ]
                else:
                    self.ensemble_weight = ensemble_weight
                
                self.ensemble_weight = np.array(self.ensemble_weight)
                self.ensemble_weight = self.ensemble_weight/sum(self.ensemble_weight)
                
                for i in range(len(self.ensemble_weight)):
                    ensempre += self.ensemble_weight[i]*pred[i]

                self.ensemble_prediction =  ensempre
                
                report = self.recall_report(self.ensemble_prediction)
                print('ensemble score is:',roc_auc_score(self.ytrue, self.ensemble_prediction))
                print('trainning report:\n',report)

        else:
            if ensemble_weight != None and len(ensemble_weight) != 4:
                    print("please enter the wrong dimension of ensemble_weight")
                    return None
            else:
                lrpred  = self.lr_cv_results.select('probability')
                dtpred  = self.dt_cv_results.select('probability')
                nbpred  = self.nb_cv_results.select('probability')
                mlppred = self.mlp_cv_results.select('probability')
                
                self.lrpred = np.array(lrpred.toPandas().probability.tolist())[:,1]
                self.dtpred = np.array(dtpred.toPandas().probability.tolist())[:,1]
                self.nbpred = np.array(nbpred.toPandas().probability.tolist())[:,1]
                self.mlppred = np.array(mlppred.toPandas().probability.tolist())[:,1]

                pred = [self.lrpred.reshape(-1,1),
                        self.dtpred.reshape(-1,1),
                        self.nbpred.reshape(-1,1),
                        self.mlppred.reshape(-1,1)
                       ]

                if ensemble_weight == None:
                    self.ensemble_weight = [
                                         1*self.lrscore, 1*self.dtscore,1*self.nbscore, 1*self.mlpscore
                                  ]
                else:
                    self.ensemble_weight = ensemble_weight
                
                self.ensemble_weight = np.array(self.ensemble_weight)
                self.ensemble_weight = self.ensemble_weight/sum(self.ensemble_weight)
                
                for i in range(len(self.ensemble_weight)):
                    ensempre += self.ensemble_weight[i]*pred[i]

                self.ensemble_prediction =  ensempre
                
                report = self.recall_report(self.ensemble_prediction)
                print('ensemble score is:',roc_auc_score(self.ytrue, self.ensemble_prediction))
                print('trainning report:\n',report)
        
        
    def recall_report(self, result, test_mode = False):
        """get report of the prediction
        """
        if test_mode:
            ppred = result.select('probability')
            ppred = np.array(ppred.toPandas().probability.tolist())[:,1]
        else:
            ppred = result
            
        ptrue = self.ytrue
        classreport = classification_report(
                            np.around(ptrue,0),
                            np.around(ppred,0),
                            target_names=['0','1'])
        
        #print(classreport)                   
        return classreport
        
        
    def ensemble_test(self):
        """get blending prediction of test data
        """
        ensempre = 0
        self.lr_test  = self.lrModel.transform(self.test)
        self.dt_test  = self.dtModel.transform(self.test)
        self.nb_test  = self.nbModel.transform(self.test)
        self.mlp_test = self.mlpModel.transform(self.test)

        self.lr_testpred  = self.lr_test.select('probability')
        self.dt_testpred  = self.dt_test.select('probability')
        self.nb_testpred  = self.nb_test.select('probability')
        self.mlp_testpred = self.mlp_test.select('probability')

        self.lr_testpred  = np.array(self.lr_testpred.toPandas().probability.tolist())[:,1]
        self.dt_testpred  = np.array(self.dt_testpred.toPandas().probability.tolist())[:,1]
        self.nb_testpred  = np.array(self.nb_testpred.toPandas().probability.tolist())[:,1]
        self.mlp_testpred = np.array(self.mlp_testpred.toPandas().probability.tolist())[:,1]

        if self.mode == 'full' and self.rf_train == True:
            self.rf_test  = self.rfModel.transform(self.test)
            self.rf_testpred  = self.rf_test.select('probability')
            self.rf_testpred  = np.array(self.rf_testpred.toPandas().probability.tolist())[:,1]

            pred = [self.lr_testpred.reshape(-1,1),
                    self.dt_testpred.reshape(-1,1),
                    self.nb_testpred.reshape(-1,1),
                    self.mlp_testpred.reshape(-1,1),
                    self.rf_testpred.reshape(-1,1)
                   ]

        else:
            pred = [self.lr_testpred.reshape(-1,1),
                    self.dt_testpred.reshape(-1,1),
                    self.nb_testpred.reshape(-1,1),
                    self.mlp_testpred.reshape(-1,1),
                   ]
            
        print("transform done!")
        self.ensemble_weight = np.array(self.ensemble_weight)
        self.ensemble_weight = self.ensemble_weight/sum(self.ensemble_weight)
        
        for i in range(len(self.ensemble_weight)):
            ensempre += self.ensemble_weight[i]*pred[i]
       
        self.test_prediction = ensempre


    

    def return_model_testresult(self, model = 'mlp', spark_mode = False):
        """get prediction of base leaner
        """
        if model == 'lr':
            obj.lr_test = obj.lrModel.transform(obj.test)
            df_get_prediction = obj.lr_test.select('prediction')
            df_get_probability = obj.lr_test.select('probability')    

        elif model == 'dt':
            obj.dt_test = obj.dtModel.transform(obj.test)
            df_get_prediction = obj.dt_test.select('prediction')
            df_get_probability = obj.dt_test.select('probability')

        elif model == 'nb':
            obj.nb_test = obj.nbModel.transform(obj.test)
            df_get_prediction = obj.nb_test.select('prediction')
            df_get_probability = obj.nb_test.select('probability')

        elif model == 'mlp':
            obj.mlp_test = obj.mlpModel.transform(obj.test)
            df_get_prediction = obj.mlp_test.select('prediction')
            df_get_probability = obj.mlp_test.select('probability')

        elif model == 'rf':
            obj.rf_test = obj.rfModel.transform(obj.test)
            df_get_prediction = obj.rf_test.select('prediction')
            df_get_probability = obj.rf_test.select('probability')

        df_get_user = obj.user_pin

        df_get_user = df_get_user.withColumn('id', fn.monotonically_increasing_id())
        df_get_prediction = df_get_prediction.withColumn('id', fn.monotonically_increasing_id()) 
        df_get_probability = df_get_probability.withColumn('id', fn.monotonically_increasing_id()) 

        df_getpro = df_get_user.join(df_get_probability,["id"], 'leftouter')
        df_getpro = df_getpro.join(df_get_prediction,["id"], 'leftouter')

        df_getpro_all = df_getpro.select('user_pin', 'probability', 'prediction')
        df_getpro = df_getpro_all.filter(df_getpro['prediction'] == 1)

        df_getpro = df_getpro.select('user_pin', 'probability')
        df_getpro = df_getpro.toPandas()

        target = np.array(df_getpro.probability.tolist())
        df_getpro['pro'] = pd.Series(target[:,1])

        df_1 = df_getpro[df_getpro.pro >= 0.8]
        print('the length of high group:',len(df_1))
        df_2 = df_getpro[((df_getpro['pro'] >= 0.7) & (df_getpro['pro'] < 0.8))]
        print('the length of medium group:',len(df_2))
        df_3 = df_getpro[((df_getpro['pro'] >= 0.5) & (df_getpro['pro'] < 0.7))]
        print('the length of low group:',len(df_3))

        if spark_mode == 'True':
            df_1_spark = spark.createDataFrame(df_1[['user_pin']])
            df_2_spark = spark.createDataFrame(df_2[['user_pin']])
            df_3_spark = spark.createDataFrame(df_3[['user_pin']])

            return df_1_spark, df_2_spark, df_3_spark, df_getpro_all, df_getpro_all
        else:
            return df_1, df_2, df_3, df_getpro

        
        
    def output(self, is_user = False):
        """get blending prediction of test
        """
        user_pin = self.user_pin
        print('done!')
        spark_pro = spark.createDataFrame(pd.DataFrame(
            {
                'pro':self.test_prediction.reshape(-1)
            },
            index = range(len(self.test_prediction))
        ))
        print('done!')
        user_pin  = user_pin.withColumn('id', fn.monotonically_increasing_id())
        spark_pro = spark_pro.withColumn('id', fn.monotonically_increasing_id())
        self.df_output = user_pin.join(spark_pro,["id"], 'leftouter')
        if is_user:
            return self.df_output.select('user_pin')
        else:
            return self.df_output.select('user_pin', 'pro')
    
    
    
    def save_document(self, df_Get, save_url, is_spark = False):
        """save result of target users
        """
        if is_spark:
            df_Get.select('user_pin').write.csv(save_url, header=True, mode='overwrite')
        else:
            df_Get = spark.createDataFrame(df_Get[['user_pin']])
            df_Get.write.csv(save_url, header=True, mode='overwrite')
        print('save done!')
        
        
"""
Reference        
[1] http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#
[2] http://spark.apache.org/docs/latest/ml-guide.html

"""

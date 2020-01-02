#********************************************************************
#Author       :   chenxinye
#Environment  :   Python3, PySpark2.4.3
#Date         :   2019-11-06

#Contact      : cclcquant@yahoo.com && https://github.com/chenxinye, if you have any suggestion, just be free to contact me. Thank you!
#*******************************************************************


import time
import gc
from sklearn.metrics import roc_auc_score
from pyspark.sql.functions import lit
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from sklearn.metrics import classification_report
import pyspark.ml.evaluation as ev
import pyspark.ml.classification as cl
from pyspark.ml import Pipeline
import pyspark.ml.feature as ft
import pyspark.ml.tuning as tune


#=============================load Data=============================
source = {
'音频目标特征配置':
's3a://f655077a6993bf0feddbb4985c0ce8ea/readonly/hive/crowd_feature_9n_82f457a2007011ea80d6fa163e83229f_20191106165642',
'音频正样本特征配置':
's3a://f655077a6993bf0feddbb4985c0ce8ea/readonly/hive/crowd_feature_9n_675f54b8006811eaba8efa163e83229f_20191106155143',
'音频负样本特征配置':
's3a://f655077a6993bf0feddbb4985c0ce8ea/readonly/hive/crowd_feature_9n_73d4f1ee006811ea808ffa163e83229f_20191106155532'
}


#==========================Preprocessing=============================
dfp = sqlContext.read.format(
        'com.databricks.spark.csv'
).options(header='True', inferschema='True').load(source['音频正样本特征配置'])

dfn = sqlContext.read.format(
        'com.databricks.spark.csv'
).options(header='True', inferschema='True').load(source['音频负样本特征配置'])

dftest = sqlContext.read.format(
        'com.databricks.spark.csv'
).options(header='True', inferschema='True').load(source['音频目标特征配置'])


add_col = (len(dftest.columns) - len(dfn.columns))

condition = ''
for j in dftest.columns[-add_col:]:
    condition += j + '=0 and '
    
dftest = dftest.filter(condition[:-4])

for j in dftest.columns[-add_col:]:
    dftest = dftest.drop(j)

    
print("The length of test columns:",len(dftest.columns))
print('positive num:', dfp.count())
print('negative num:', dfn.count())
print('test num:', dftest.count())





#=========================Feature engineering==============================


def drop_stdzero(dfp, dfn):
    dfp_s = dfp.sample(False, 0.1, 2019).toPandas()
    dfn_s = dfn.sample(False, 0.1, 2019).toPandas()
    
    train_std = pd.concat([dfp_s,dfn_s],axis = 0).std()
    stdzero_col = train_std[train_std==0].index.tolist()
    for i in stdzero_col:
        dfp = dfp.drop(i)
        dfn = dfn.drop(i)
    return dfp, dfn, stdzero_col



def check_null(df):
    if len(df.columns) == 0 or len(df.dropna()) == 0:
        print("none!")
        return False
    else:
        return True


def list_minus(list1,list2):
    return [i for i in list1 if i not in list2]




def fe_generate(data,ret = True):
    data = data.toPandas()
    col_all = data.columns

    ##1.features of buying behavior
    # pchs_firstcate_columns
    pchs_firstcate_columns = [col for col in data.columns if 'pchs' in col and 'secondcate' not in col and 'attr' not in col]
    #print(pchs_firstcate_columns)

    # pchs_secondcate_columns
    pchs_secondcate_columns = [col for col in data.columns if 'pchs' in col and 'secondcate' in col]
    #print(pchs_secondcate_columns)

    # pchs_columns
    pchs_columns = [col for col in data.columns if 'pchs' in col]
    #print(pchs_columns)

    ##2.features of following behaviour
    flw_secondcate_columns = [col for col in data.columns if 'flw' in col]
    #print(flw_secondcate_columns)

    ##3.features of browser behaviour
    brs_secondcate_columns = [col for col in data.columns if 'brs' in col and 'secondcate' in col]
    #print(brs_secondcate_columns)

    brs_firstcate_columns = [col for col in data.columns if 'brs' in col and 'secondcate' not in col]
    #print(brs_firstcate_columns)

    brs_columns = [col for col in data.columns if 'brs' in col]
    #print(brs_columns)

    ##4.features of ad click
    clk_secondcate_columns = [col for col in data.columns if 'clk' in col and 'secondcate' in col]
    #print(clk_secondcate_columns)

    clk_firstcate_columns = [col for col in data.columns if 'clk' in col and 'secondcate' not in col]
    #print(clk_firstcate_columns)

    clk_columns = [col for col in data.columns if 'clk' in col]
    #print(clk_columns)

    ##5.features of user attribute 
    attr_columns = [col for col in data.columns if 'attr' in col]
    #print(attr_columns)

    ##6.features of add repeat buying behavior
    cart_secondcate_columns = [col for col in data.columns if 'cart' in col and 'secondcate' in col]
    #print(cart_secondcate_columns)

    cart_firstcate_columns = [col for col in data.columns if 'cart' in col and 'secondcate' not in col]
    #print(cart_firstcate_columns)

    cart_columns = [col for col in data.columns if 'cart' in col]
    #print(cart_columns)

    columns_dict = {
        'pchs_firstcate_columns':pchs_firstcate_columns,
        'pchs_secondcate_columns':pchs_secondcate_columns,
        'pchs_columns':pchs_columns,
        'flw_secondcate_columns':flw_secondcate_columns,
        'brs_secondcate_columns':brs_secondcate_columns,
        'brs_firstcate_columns':brs_firstcate_columns,
        'brs_columns':brs_columns,
        'clk_firstcate_columns':clk_firstcate_columns,
        'clk_secondcate_columns':clk_secondcate_columns,
        'clk_columns':clk_columns,
        'attr_columns':attr_columns,
        'cart_secondcate_columns':cart_secondcate_columns,
        'cart_firstcate_columns':cart_firstcate_columns,
        'cart_columns':cart_columns
    }


    attrlist = list()
    for i in columns_dict:
        attrlist += columns_dict[i]

    col_reduce = list_minus(col_all,attrlist)

    ###########################transformation##########################
    attr_product = data[attr_columns]

    pchs_firstcate_product = data[pchs_firstcate_columns]
    pchs_secondcate_product = data[pchs_secondcate_columns]
    pchs_product = data[pchs_columns]

    brs_firstcate_product = data[brs_firstcate_columns]
    brs_secondcate_product = data[brs_secondcate_columns]
    brs_product = data[brs_columns]

    clk_firstcate_product = data[clk_firstcate_columns]
    clk_secondcate_product = data[clk_secondcate_columns]
    clk_product = data[clk_columns]

    cart_firstcate_product = data[cart_firstcate_columns]
    cart_secondcate_product = data[cart_secondcate_columns]
    cart_product = data[cart_columns]

    user_attr_all_sum = pd.DataFrame({'attr_all_sum':attr_product.sum(axis = 1)})
    user_pchs = pd.DataFrame({'pchs_firstcate_sum':pchs_firstcate_product.sum(axis=1),
                              'pchs_secondcate_sum':pchs_secondcate_product.sum(axis=1)
                             })

    user_pchs_all_sum = pd.DataFrame({'pchs_all_sum':pchs_product.sum(axis=1)})
    pchs_firstcate_sum = pd.DataFrame({'pchs_firstcate_sum':pchs_firstcate_product.sum(axis=1)})
    pchs_secondcate_sum = pd.DataFrame({'pchs_secondcate_sum':pchs_secondcate_product.sum(axis=1)})


    user_brs = pd.DataFrame({'brs_firstcate_sum':brs_firstcate_product.sum(axis=1),
                             'brs_secondcate_sum':brs_secondcate_product.sum(axis=1)
                            })

    user_brs_all_sum = pd.DataFrame({'brs_all_sum':brs_product.sum(axis = 1)})
    brs_firstcate_sum = pd.DataFrame({'brs_firstcate_sum':brs_firstcate_product.sum(axis=1)})
    brs_secondcate_sum = pd.DataFrame({'brs_secondcate_sum':brs_secondcate_product.sum(axis=1)})  


    user_clk = pd.DataFrame({'clk_firstcate_sum':clk_firstcate_product.sum(axis=1),
                             'clk_secondcate_sum':clk_secondcate_product.sum(axis=1)
                            })

    user_clk_all_sum = pd.DataFrame({'clk_all_sum':clk_product.sum(axis=1)})
    clk_firstcate_sum = pd.DataFrame({'clk_firstcate_sum':clk_firstcate_product.sum(axis=1)})
    clk_secondcate_sum = pd.DataFrame({'clk_secondcate_sum':clk_secondcate_product.sum(axis=1)})


    user_cart = pd.DataFrame({'cart_firstcate_sum':cart_firstcate_product.sum(axis=1),
                              'cart_secondcate_sum':cart_secondcate_product.sum(axis=1)
                             })

    user_cart_all_sum = pd.DataFrame({'cart_all_sum':cart_product.sum(axis=1)})
    cart_firstcate_sum = pd.DataFrame({'cart_firstcate_sum':cart_firstcate_product.sum(axis=1)})
    cart_secondcate_sum = pd.DataFrame({'cart_secondcate_sum':cart_secondcate_product.sum(axis=1)})


    user_rate = pd.DataFrame({
        'attr_sum':user_attr_all_sum.attr_all_sum,
        'pchs_sum':user_pchs_all_sum.pchs_all_sum,
        'clk_sum':user_clk_all_sum.clk_all_sum,
        'brs_sum':user_brs_all_sum.brs_all_sum,
        'cart_sum':user_cart_all_sum.cart_all_sum,
        'clk_rate':user_pchs_all_sum.pchs_all_sum/user_clk_all_sum.clk_all_sum,
        'clk_firstcate_rate':user_pchs.pchs_firstcate_sum/user_clk.clk_firstcate_sum,
        'clk_secondcate_rate':user_pchs.pchs_secondcate_sum/user_clk.clk_secondcate_sum,
        'brs_rate':user_pchs_all_sum.pchs_all_sum/user_brs_all_sum.brs_all_sum,
        'brs_firstcate_rate':user_pchs.pchs_firstcate_sum/user_brs.brs_firstcate_sum,
        'brs_secondcate_rate':user_pchs.pchs_secondcate_sum/user_brs.brs_secondcate_sum,
        'cart_rate':user_pchs_all_sum.pchs_all_sum/user_cart_all_sum.cart_all_sum,
        'cart_firstcate_rate':user_pchs.pchs_firstcate_sum/user_cart.cart_firstcate_sum,
        'cart_secondcate_rate':user_pchs.pchs_secondcate_sum/user_cart.cart_secondcate_sum,
        'attr_rate':user_pchs_all_sum.pchs_all_sum/user_attr_all_sum.attr_all_sum
    })

    if check_null(user_rate.dropna(how = 'all',axis = 1)):
        print("data has been added into user_rate!")
        user_rate.dropna(how = 'all', axis = 1, inplace = True)
        user_rate[np.isinf(user_rate)] = -1
        data = pd.concat([data, user_rate], axis = 1).fillna(0)
    else:
        print("user_rate is not installded")
    
    if ret:
        return data, columns_dict, col_reduce, user_rate
    else:
        return data



dfp, dfn, dropcol = drop_stdzero(dfp,dfn)

dfp = fe_generate(dfp, False)
dfn = fe_generate(dfn, False)
dftest = fe_generate(dftest, False)

dfp = spark.createDataFrame(dfp)
dfn = spark.createDataFrame(dfn)
dftest = spark.createDataFrame(dftest)




#============================Training begin=============================
class spark_mlpipeline:

    #Parameter:
    #::dfp,dfn: the positive dataset and negative dataset respectively
    #::weight: the proportion of train and test respectively
    #::ensemble_weight: the weight to each model prediction. eg: sometimes, if the model performs better, we give it highter weight
    #::test: choose the target dataset(without label) 
    #::alpha: choose the proportion of the best trainning features, the parameter is an element of [0,1]
    #::mode: the parameter means whether you want to train hight accuracy model even if it will take more time
    #::verbose: silent or print trainning information
    #::seed: set random seed to data and model trainning 

    def __init__ (self, dfp, dfn, weight = [4.0,1.0],
                  ensemble_weight = None,
                  test = None, alpha = 0.5, mode = 'full', 
                  verbose = True, seed = 2020):

        self.seed = seed
        self.weight = weight
        self.mode = mode
        self.alpha = alpha
        self.verbose = verbose
        self.ensemble_weight = ensemble_weight
        
        dfp = dfp.withColumn('label',lit(1))
        dfn = dfn.withColumn('label',lit(0))
        
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
        
        p_count = dfp.count()
        n_count = dfn.count()
        
        if n_count <= p_count:
            dfp = dfp.sample(False, n_count/p_count, self.seed)
        else:
            dfn = dfn.sample(False, p_count/n_count, self.seed)
            
        return dfp, dfn
        
        
    def Fe_gneer(self):
        
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
        
        if self.mode == 'fast':
            _iter = 20
            _regParam = [0.11,0.21]
            
        elif self.mode == 'full':
            _iter = round(len(self.fe_col)*0.3)
            _regParam = [0.01,0.05,0.1,0.15,0.3]
            
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
        
        if self.mode == 'fast':
            _maxDepth = [round(i*len(self.fe_col)) for i in [0.07]]
            
        elif self.mode == 'full':
            _maxDepth = [round(i*len(self.fe_col)) for i in [0.05,0.06,0.7]]
            
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
        
        mlp = MultilayerPerceptronClassifier(
            maxIter=60, 
            layers=[int(self.alpha*len(self.fe_col)), 80, 80, 80, 2], 
            labelCol = 'label',
            featuresCol = 'features',
            blockSize=300,
            seed=self.seed
        )

        self.mlpModel = mlp.fit(self.train_data)
        self.mlp_cv_results = self.mlpModel.transform(self.test_data)
        self.mlpscore = self.evaluator.evaluate(self.mlp_cv_results, {self.evaluator.metricName: 'areaUnderROC'})
        
        if self.verbose:
            print("AUC score is:", self.mlpscore)
            #print("Area Under PR is:",self.evaluator.evaluate(self.mlp_cv_results, {self.evaluator.metricName: 'areaUnderPR'}))
   
        pass


    def rf_cv(self):
        
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
            print('randomforest')
            bt = time.time()
            self.rf_cv()
            et = time.time()
            print('rf time cost: {}'.format(et - bt))
            
            del bt, et; gc.collect()
        
        print("training done!")
        print("\n-----------------finish-----------------\n")
        
        
    def ensemble(self, ensemble_weight = None):
        ensempre = 0
        if self.mode == 'full':
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

                for i in range(len(self.weight)):
                    ensempre += self.ensemble_weight[i]*pred[i]

                self.ensemble_prediction = (ensempre - ensempre.min())/(ensempre.max() - ensempre.min())
                
                report = self.recall_report(self.ensemble_prediction)
                print('ensemble score is:',roc_auc_score(self.ytrue, self.ensemble_prediction))
                print('trainning report:\n',report)

        elif self.mode == 'fast':
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

                for i in range(len(self.weight)):
                    ensempre += self.ensemble_weight[i]*pred[i]

                self.ensemble_prediction = (ensempre - ensempre.min())/(ensempre.max() - ensempre.min())
                
                report = self.recall_report(self.ensemble_prediction)
                print('ensemble score is:',roc_auc_score(self.ytrue, self.ensemble_prediction))
                print('trainning report:\n',report)
        
        
    def recall_report(self, result, test_mode = False):
        
        if test_mode:
            ptrue = result.select('probability')
            ptrue = np.array(ptrue.toPandas().probability.tolist())[:,1]
        else:
            ptrue = result
            
        ppred = self.ytrue
        classreport = classification_report(
                            np.around(ptrue,0),
                            np.around(ppred,0),
                            target_names=['0','1'])
        
        #print(classreport)                   
        return classreport
        
        
    def ensemble_test(self):
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

        if self.mode == 'full':
            self.rf_test  = self.rfModel.transform(self.test)
            self.rf_testpred  = self.rf_test.select('probability')
            self.rf_testpred  = np.array(self.rf_testpred.toPandas().probability.tolist())[:,1]

            pred = [self.lr_testpred.reshape(-1,1),
                    self.dt_testpred.reshape(-1,1),
                    self.nb_testpred.reshape(-1,1),
                    self.mlp_testpred.reshape(-1,1),
                    self.rf_testpred.reshape(-1,1)
                   ]

        elif self.mode == 'fast':
            pred = [self.lr_testpred.reshape(-1,1),
                    self.dt_testpred.reshape(-1,1),
                    self.nb_testpred.reshape(-1,1),
                    self.mlp_testpred.reshape(-1,1),
                   ]

        for i in range(len(self.ensemble_weight)):
            ensempre += self.ensemble_weight[i]*pred[i]

        self.test_prediction = (ensempre - ensempre.min())/(ensempre.max() - ensempre.min())

        
    def output(self):
        user_pin = self.user_pin.toPandas().user_pin
        self.df_output = pd.DataFrame(columns = {'probability','user_pin'})
        self.df_output.probability = pd.Series(self.test_prediction.reshape(-1))
        self.df_output.user_pin = user_pin
        return self.df_output


obj  = spark_mlpipeline(dfp, dfn, weight = [4.0,1.0],
                  test = dftest, alpha = 0.5, mode = 'fast', 
                  verbose = True, seed = 42
)


obj.train_all()
obj.ensemble()
obj.ensemble_test()


#================================Get result==========================

df_Get = obj.output()
df_get_ = df_Get[df_Get.probability > 0.59][['user_pin']]
print(len(df_get))
df_get = spark.createDataFrame(df_get)


#==================================Save==============================

save_url = 's3a://f655077a6993bf0feddbb4985c0ce8ea/datasets/save/f00b2e60ad1911e9a965fa163e70e262/dt_video0.59.csv'
df_get.write.csv(save_url, header=True, mode='overwrite')



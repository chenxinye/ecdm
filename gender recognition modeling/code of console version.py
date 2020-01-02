#############################################validation for gender recognition modelling################################################

# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:12:43 2019
@author: chenxinye
@title: modelling for gender recognition
@environment: python 3; pyspark 2.1.2; jupyter notebook of jiushu Platform of JD(京东九数)
"""

#The pyspark library has been imported internally
import re
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn import preprocessing
import numpy as np
import pandas as pd
import pandas as pd
from pyspark.sql.functions import lit
import warnings
warnings.filterwarnings('ignore')
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

data = {
"男性强特":
'/user/f655077a6993bf0feddbb4985c0ce8ea/datasets/hive/crowd_feature_9n_90d5b36cebc011e9955efa163e863935_20191011',
"女性强特":
'/user/f655077a6993bf0feddbb4985c0ce8ea/datasets/hive/crowd_feature_9n_80e7c3d2ebc011e98805fa163e863935_20191011',
"美妆特征":
'/user/f655077a6993bf0feddbb4985c0ce8ea/datasets/hive/crowd_feature_9n_8ee7f15aec9211e9955efa163e863935_20191012'
}


com = 'com.databricks.spark.csv'

#dataset for training and testing 
df_male = sqlContext.read.format(
    com).options(
    header='True', inferschema='True').load(data['男性强特']).sample(False,0.005,42)

df_male = df_male.withColumn('target', lit(1))

df_female = sqlContext.read.format(
    com).options(
    header='True', inferschema='True').load(data['女性强特']).sample(False,0.005,42)

df_female = df_female.withColumn('target', lit(0))

#selected features
chfe = [
        "爱下厨偏好未知",
        "爱下厨偏好极度爱好",
        "爱下厨偏好高度爱好",
        "爱下厨偏好中度爱好",
        "爱下厨偏好轻度爱好",
        "女装用户未知",
        "女装用户精确",
        "女装用户广泛",
        "孕龄未知",
        "怀孕妈妈",
        "孩子0-3个月",
        "孩子3-6个月",
        "孩子6-12个月",
        "孩子1-3岁",
        "孩子3-6岁",
        "孩子6岁以上",
        "关注化妆品未知",
        "关注化妆品精确",
        "关注化妆品广泛",
        "关注女装用户未知",
        "关注女装用户精确",
        "关注女装用户广泛",
        "高度关注女装用户未知",
        "高度关注女装用户精确",
        "高度关注女装用户广泛",
        "高度关注男装用户未知",
        "高度关注男装用户精确",
        "高度关注男装用户广泛",
        "母婴",
        "珠宝首饰",
        "男装",
        "女装",
        "男士面部护肤",
        "美妆工具",
        "武术搏击",
        "育儿/家教",
        "时尚/美妆",
        "婚恋与两性",
        "孕产/胎教",
        "美发假发/造型",
        "女性护理",
        "文胸",
        "吊带/背心",
        "女式内裤",
        "男式内裤",
        "商务男袜",
        "抹胸",
        "文胸套装",
        "少女文胸",
        "女士丝巾/围巾/披肩",
        "男士丝巾/围巾",
        "男士彩妆",
        "点击近一个月胸罩广告",
        "点击近一个月仿真阳具广告",
        "点击近一个月避孕套广告",
        "点击近一个月吊带广告",
        "点击近一个月女性励志书籍广告",
        "近六个月搜索关键词女",
        "近六个月搜索关键词男",
        "近六个月搜索关键词搏击",
        "近六个月搜索关键词美妆",
        "半年内购买避孕套",
        "半年内排卵验孕类目",
        "半年内男用延时",
        "半年内购买充气/仿真娃娃",
        "半年内购买妈妈专区",
        "半年内购买精品男包",
        "半年内购买潮流女包",
        "半年内购买男士丝巾/围巾",
        "半年内购买男士面部护肤",
        "近一年内购买充气/仿真娃娃",
        "近一年内购买火机烟具",
        "近一年内购买流行男鞋",
        "近一年内购买时尚女鞋",
        "近一年内购买精品男包",
        "近一年购买潮流女包"
       ]


df = df_male.union(df_female)
#dfsample = df.sample(False, 0.005, 2019)
#since the total number of the dataset is too big, it'll take a lot of time
print("测试样本总数：", df.count())

def _map_to_pandas(rdds):
    return [pd.DataFrame(list(rdds))]

def toPandas(df, n_partitions=None):
    if n_partitions is not None: df = df.repartition(n_partitions)
    df_pand = df.rdd.mapPartitions(_map_to_pandas).collect()
    df_pand = pd.concat(df_pand)
    df_pand.columns = df.columns
    return df_pand

dfsample = toPandas(df)

if len(dfsample.columns) == (len(chfe) + 1 + 1):
    for i in [i + 1 for i in range(len(dfsample.columns) - 2)]:
        print(dfsample.columns[i], ':', chfe[i - 1])
        dfsample = dfsample.rename(columns = {dfsample.columns[i]:chfe[i - 1]})
        
#%matplotlib inline
matplotlib.use('qt4agg')
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family']='sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False

_dfsample = dfsample.drop(["user_pin","target"],axis = 1)

fe_corr = _dfsample.corr()

#print(fe_corr)

plt.figure(figsize=(15,10))
sns.heatmap(fe_corr, cmap='RdBu_r', center=0.0) 
plt.title('Correlations',fontsize=16)
plt.show()

corrs = fe_corr.abs().unstack().sort_values(kind="quicksort",ascending = False).dropna(how = 'any').reset_index()
corrs = corrs[corrs['level_0'] != corrs['level_1']].reset_index()
corrs = corrs.rename(columns = {0:'corr_value'})
corrs.head(30)

#----------------------------------output--------------------------------------
#index	level_0	level_1	corr_value
#0	75	高度关注女装用户广泛	高度关注女装用户未知	0.934063
#1	76	高度关注女装用户未知	高度关注女装用户广泛	0.934063
#2	77	女装用户未知	女装用户广泛	0.914450
#3	78	女装用户广泛	女装用户未知	0.914450
#4	79	男士面部护肤	半年内购买男士面部护肤	0.755487
#5	80	半年内购买男士面部护肤	男士面部护肤	0.755487
#6	81	高度关注男装用户未知	高度关注男装用户精确	0.745259
#7	82	高度关注男装用户精确	高度关注男装用户未知	0.745259
#8	83	孩子1-3岁	孕龄未知	0.739289
#9	84	孕龄未知	孩子1-3岁	0.739289
#10	85	近一年内购买充气/仿真娃娃	半年内购买充气/仿真娃娃	0.732682
#11	86	半年内购买充气/仿真娃娃	近一年内购买充气/仿真娃娃	0.732682
#12	87	关注女装用户未知	关注女装用户广泛	0.726467
#13	88	关注女装用户广泛	关注女装用户未知	0.726467
#14	89	半年内购买精品男包	近一年内购买精品男包	0.725556
#15	90	近一年内购买精品男包	半年内购买精品男包	0.725556
#16	91	关注化妆品广泛	关注化妆品未知	0.724229
#17	92	关注化妆品未知	关注化妆品广泛	0.724229
#18	93	近一年购买潮流女包	半年内购买潮流女包	0.708727
#19	94	半年内购买潮流女包	近一年购买潮流女包	0.708727
#20	95	关注女装用户精确	关注女装用户未知	0.685228
#21	96	关注女装用户未知	关注女装用户精确	0.685228
#22	97	关注化妆品未知	关注化妆品精确	0.683889
#23	98	关注化妆品精确	关注化妆品未知	0.683889
#24	99	高度关注男装用户广泛	高度关注男装用户未知	0.663156
#25	100	高度关注男装用户未知	高度关注男装用户广泛	0.663156
#26	101	爱下厨偏好未知	爱下厨偏好高度爱好	0.533296
#27	102	爱下厨偏好高度爱好	爱下厨偏好未知	0.533296
#28	103	爱下厨偏好未知	爱下厨偏好轻度爱好	0.500022
#29	104	爱下厨偏好轻度爱好	爱下厨偏好未知	0.500022

"""
col1, col2, cor = [],[],[]
columns = np.full((fe_corr.shape[0],), True, dtype=bool)
for i in range(fe_corr.shape[0]):
    for j in range(i+1, fe_corr.shape[0]):
        if fe_corr.iloc[i,j] >= 0.005:
            #print(fe_corr.columns[i],fe_corr.columns[j],fe_corr.iloc[i,j])
            col1.append(fe_corr.columns[i]);
            col2.append(fe_corr.columns[j]);
            cor.append(fe_corr.iloc[i,j])
            
            if columns[j]:
                columns[j] = False
                
#selected_columns = dfsample.columns[columns]
#selected_columns.shape
"""

positie_dict_corr = pd.DataFrame({'id1':corrs.level_0,'id2':corrs.level_1,'corr':corrs.corr_value})
print(positie_dict_corr)

#positie_dict_corr = positie_dict_corr.sort_values(by = 'corr', ascending = False).reset_index(drop = True)

weight = dfsample.columns



mode = 'MULTIPLY' # 'MULTIPLY' / 'ADD'

if mode == 'ADD':
    weinum = [0]*len(dfsample.columns)
    weightdict = dict(zip(weight,weinum))
    for i in range(len(positie_dict_corr)):
        if positie_dict_corr.loc[i][1].find('男') != -1 and positie_dict_corr.loc[i][2].find('女') != -1:
            weightdict[positie_dict_corr.iloc[i][1]] += positie_dict_corr.iloc[i][0] * 40
            weightdict[positie_dict_corr.iloc[i][2]] += positie_dict_corr.iloc[i][0] * -40

        elif positie_dict_corr.loc[i][1].find('女') != -1 and positie_dict_corr.loc[i][2].find('男') != -1:
            weightdict[positie_dict_corr.iloc[i][1]] += positie_dict_corr.iloc[i][0] * -40
            weightdict[positie_dict_corr.iloc[i][2]] += positie_dict_corr.iloc[i][0] * 40

        elif positie_dict_corr.loc[i][1].find('男') != -1 and positie_dict_corr.loc[i][2].find('女') == -1:
            weightdict[positie_dict_corr.iloc[i][1]] += positie_dict_corr.iloc[i][0] * 70
            weightdict[positie_dict_corr.iloc[i][2]] += positie_dict_corr.iloc[i][0] * 40

        elif positie_dict_corr.loc[i][1].find('女') != -1 and positie_dict_corr.loc[i][2].find('男') == -1:
            weightdict[positie_dict_corr.iloc[i][1]] += positie_dict_corr.iloc[i][0] * -60
            weightdict[positie_dict_corr.iloc[i][2]] += positie_dict_corr.iloc[i][0] * -40

        elif positie_dict_corr.loc[i][1].find('男') == -1 and positie_dict_corr.loc[i][2].find('女') != -1:
            weightdict[positie_dict_corr.iloc[i][1]] += positie_dict_corr.iloc[i][0] * -40
            weightdict[positie_dict_corr.iloc[i][2]] += positie_dict_corr.iloc[i][0] * -60

        elif positie_dict_corr.loc[i][1].find('女') == -1 and positie_dict_corr.loc[i][2].find('男') != -1:
            weightdict[positie_dict_corr.iloc[i][1]] += positie_dict_corr.iloc[i][0] * 40
            weightdict[positie_dict_corr.iloc[i][2]] += positie_dict_corr.iloc[i][0] * 70

        else:pass
        
elif mode == 'MULTIPLY':
    weinum = [1]*len(dfsample.columns)
    weightdict = dict(zip(weight,weinum))
    for i in range(len(positie_dict_corr)):
        if positie_dict_corr.iloc[i][1].find('男') != -1 and positie_dict_corr.iloc[i][2].find('女') != -1:
            weightdict[positie_dict_corr.iloc[i][1]] = \
            abs(weightdict[positie_dict_corr.iloc[i][1]]*(1 + positie_dict_corr.iloc[i][0]))
            weightdict[positie_dict_corr.iloc[i][2]] = -\
            abs(weightdict[positie_dict_corr.iloc[i][2]]*(1 + positie_dict_corr.iloc[i][0]))

        elif positie_dict_corr.loc[i][1].find('女') != -1 and positie_dict_corr.iloc[i][2].find('男') != -1:
            weightdict[positie_dict_corr.iloc[i][1]] = -\
            abs(weightdict[positie_dict_corr.loc[i][1]]*(1 + positie_dict_corr.iloc[i][0]))
            weightdict[positie_dict_corr.iloc[i][2]] = \
            abs(weightdict[positie_dict_corr.iloc[i][2]]*(1 + positie_dict_corr.iloc[i][0]))

        elif positie_dict_corr.iloc[i][1].find('男') != -1 and positie_dict_corr.iloc[i][2].find('女') == -1:
            weightdict[positie_dict_corr.iloc[i][1]] = \
            abs(weightdict[positie_dict_corr.iloc[i][1]]*(1 + 2*positie_dict_corr.iloc[i][0]))
            weightdict[positie_dict_corr.iloc[i][2]] = \
            abs(weightdict[positie_dict_corr.iloc[i][2]]*(1 + 0.5*positie_dict_corr.iloc[i][0]))

        elif positie_dict_corr.iloc[i][1].find('女') != -1 and positie_dict_corr.iloc[i][2].find('男') == -1:
            weightdict[positie_dict_corr.iloc[i][1]] = -\
            abs(weightdict[positie_dict_corr.iloc[i][1]]*(1 + 2*positie_dict_corr.iloc[i][0]))
            weightdict[positie_dict_corr.iloc[i][2]] = -\
            abs(weightdict[positie_dict_corr.iloc[i][2]]*(1 + 0.5*positie_dict_corr.iloc[i][0]))

        elif positie_dict_corr.loc[i][1].find('男') == -1 and positie_dict_corr.iloc[i][2].find('女') != -1:
            weightdict[positie_dict_corr.iloc[i][1]] = -\
            abs(weightdict[positie_dict_corr.iloc[i][1]]*(1 + 0.5*positie_dict_corr.iloc[i][0]))
            weightdict[positie_dict_corr.iloc[i][2]] = -\
            abs(weightdict[positie_dict_corr.iloc[i][2]]*(1 + 2*positie_dict_corr.iloc[i][0]))

        elif positie_dict_corr.loc[i][1].find('女') == -1 and positie_dict_corr.iloc[i][2].find('男') != -1:
            weightdict[positie_dict_corr.iloc[i][1]] = \
            abs(weightdict[positie_dict_corr.iloc[i][1]]*(1 + 0.5*positie_dict_corr.iloc[i][0]))
            weightdict[positie_dict_corr.iloc[i][2]] = \
            abs(weightdict[positie_dict_corr.iloc[i][2]]*(1 + 2*positie_dict_corr.iloc[i][0]))
    
        else:pass

     
for i in weightdict:
    if i != 'user_pin' and i != 'target':
        print(i, ':', weightdict[i])


name, key = list(), list()

for i in weightdict:
    name.append(name); key.append(weightdict[i])
    

#target dataset of model application
df_test = sqlContext.read.format(
    'com.databricks.spark.csv'
).options(header='True', inferschema='True').load(data['美妆特征'])

print("美妆购买人数:",df_test.count())

#df_test = df_test.toPandas()
import time
st = time.time()
df_test = toPandas(df_test)
ed = time.time()
print("time cost:{}".format(ed - st))


if len(df_test.columns) == (len(chfe) + 2):
    for i in [i + 1 for i in range(len(df_test.columns) - 2)]:
        #print(df_test.columns[i],':',chfe[i - 1])
        df_test = df_test.rename(columns = {df_test.columns[i]:chfe[i - 1]})
        

def test_weight(train,test,weightdict):
    for i in train.columns:
        if i != 'user_pin' and i != 'target':
            #print(i)
            if weightdict[i] != 0:
                train[i] = weightdict[i]*train[i]#/sum(key)
                test[i] = weightdict[i]*test[i]#/sum(key)
                #train[i] = train[i].astype('float32')
                #test[i] = test[i].astype('float32')
            else:
                train = train.drop(i, axis = 1)
                test = test.drop(i, axis = 1)
                
    return train, test
    

train, test = test_weight(dfsample, df_test, weightdict)



#stan = preprocessing.StandardScaler()
#train_scaled = stan.fit_transform(train.drop(['user_pin','target'],axis = 1))
#test_scaled = stan.transform(test.drop('user_pin',axis = 1))

train_scaled = train.drop(['user_pin','target'],axis = 1)
test_scaled = test.drop(['user_pin',test.columns[-1]],axis = 1)

train['label'] = train_scaled.sum(axis = 1)
test['label'] = test_scaled.sum(axis = 1)


fig,ax=plt.subplots()
ax.hist(train['label'], bins=100, histtype="stepfilled", normed=True, alpha=0.39)
sns.kdeplot(train['label'],shade=True)
#sns.distplot(train['label'])
plt.show()

train['label'] = (train['label'] - train['label'].min())/(train['label'].max() - train['label'].min())
test['label'] = (test['label'] - test['label'].min())/(test['label'].max() - test['label'].min())


def labeling(series, alpha = 0.5):
    series = np.array(series)
    s = pd.Series(np.zeros(len(series)))
    for i in range(len(series)):
        #print(series[i])
        if series[i] >= alpha:
            s[i] = 1
        else:
            s[i] = 0
    return s

train['label_int'] = labeling(train.label)


from sklearn.metrics import classification_report
print(classification_report(train['target'], train['label_int'], target_names = ['男','女']))

from sklearn.metrics import confusion_matrix
sns.set()
cm = confusion_matrix(train['target'], train['label_int'])
sns.heatmap(cm,annot=True)


#---------------------------------training with weighted features------------------------------------------

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from catboost import Pool, CatBoostClassifier

train_ = train_scaled
target_ = train['target'] #The target variable used is the gender attributes of user matched in the Jiushu.

model = CatBoostClassifier(
               loss_function="Logloss", 
               eval_metric="AUC", 
               learning_rate=0.01, 
               iterations=10000, 
               l2_leaf_reg=10, 
               random_seed=2019, 
               od_type="Iter", 
               depth=10, 
               early_stopping_rounds=40, 
               border_count=64
               )

n_split = 5
kf = KFold(n_splits=n_split, random_state=2022, shuffle=True)
y_valid_pred = 0 * target_
y_test_pred = 0

# When applying the mode of multiply, the rate of convergence seems get faster!
for idx, (train_index, valid_index) in enumerate(kf.split(train_)):
    y_train, y_valid = target_.iloc[train_index], target_.iloc[valid_index]
    X_train, X_valid = train_.iloc[train_index,:], train_.iloc[valid_index,:]
    _train = Pool(X_train, label=y_train)
    _valid = Pool(X_valid, label=y_valid)
    print( "\nfold ", idx)
    cat_model = model.fit(_train,
                          eval_set=_valid,
                          use_best_model=True,
                          verbose=30,
                          plot=True
                         )
    
    pred = cat_model.predict_proba(X_valid)[:,1]
    print( "auc = ", roc_auc_score(y_valid, pred))
    y_valid_pred.iloc[valid_index] = pred
    y_test_pred += cat_model.predict_proba(test_scaled)[:,1]
    
y_test_pred = y_test_pred / n_split
#test['label_int'] = np.around(y_test_pred, 0.5)


#To decide blending weight to fit business requirement.
train_pred = 0.99*y_valid_pred + 0.01*train.label   # making the train.label as kind of penalty factor
test_pred = 0.99*y_test_pred + 0.01*test.label   # making the train.label as kind of penalty factor

print(classification_report(train['target'], np.around(train_pred, 0), target_names = ['男','女']))
print("auc score:", roc_auc_score(train['target'],train_pred))

sns.set()
cm = confusion_matrix(train['target'], np.around(train_pred,0))
sns.heatmap(cm,annot=True)

test['label_int'] = np.around(test_pred, 0)

dt_get = test[test['label_int'] == 0]
dt_get_user = dt_get[['user_pin']]
print("京东平台标记为女的人数占预测女性人群的占比, ",len(dt_get[dt_get.attr_cpp_base_sex_22899 == 1])/len(dt_get))
#blending weight: 0.90 0.10 , 京东平台标记为女的人数占预测女性人群的占比: 0.8113920012004365
#blending weight: 0.98 0.02 , 京东平台标记为女的人数占预测女性人群的占比: 0.8139204061945112
#blending weight: 0.99 0.01 , 京东平台标记为女的人数占预测女性人群的占比: 0.8139600126656545
#blending weight: 1.00 0.00 , 京东平台标记为女的人数占预测女性人群的占比: 0.8135517516769488

spark_df = spark.createDataFrame(dt_get_user)

target_file = "/user/f655077a6993bf0feddbb4985c0ce8ea/datasets/my_folder/girl_final.csv"
middle_dir = "/user/f655077a6993bf0feddbb4985c0ce8ea/datasets/my_folder/girl_final"

spark_df.repartition(1).write.mode("overwrite").format("com.databricks.spark.csv").save(middle_dir,header=True) 

def copyMergeFile(middle_dir, target_file): 
    URI = sc._gateway.jvm.java.net.URI 
    Path = sc._gateway.jvm.org.apache.hadoop.fs.Path
    FileSystem = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem 
    Configuration = sc._gateway.jvm.org.apache.hadoop.conf.Configuration
    FileUtil = sc._gateway.jvm.org.apache.hadoop.fs.FileUtil 
    hadoopconf = Configuration() 
    fs = FileSystem.get(URI("hdfs://pino"),hadoopconf) 
    srcDir = Path(middle_dir)
    dstFile = Path(target_file) 
    FileUtil.copyMerge(fs, srcDir, fs, dstFile, False, hadoopconf,"")

copyMergeFile(middle_dir, target_file)



#--------------------weightdict -> name, key (mode: multiply)------------------
#高度关注男装用户精确 : 33.56211063
#文胸 : -7.63048778945
#近六个月搜索关键词美妆 : 1.89187384624
#孕龄未知 : -4.07131475388
#近六个月搜索关键词男 : 80818211.2466
#点击近一个月吊带广告 : -1.19847358607
#高度关注男装用户广泛 : 34.9286988048
#男士丝巾/围巾 : 35.5250517882
#爱下厨偏好轻度爱好 : -1.7025476159
#爱下厨偏好高度爱好 : -1.88908943776
#少女文胸 : -24.3543727273
#爱下厨偏好未知 : -3.04330455467
#孩子3-6个月 : -1.31110717307
#孩子1-3岁 : -2.55910390525
#近一年购买潮流女包 : -46496.6529442
#点击近一个月避孕套广告 : -1.25000419994
#高度关注男装用户未知 : 267.690753064
#女装用户精确 : -54.8723209206
#关注化妆品广泛 : -4.0195629284
#半年内购买潮流女包 : -6881.37038063
#武术搏击 : -1.98011647092
#吊带/背心 : -3.85705390143
#关注女装用户精确 : -2520.56174462
#半年内购买妈妈专区 : -3.11306645451
#近六个月搜索关键词搏击 : -1.09411747176
#文胸套装 : -2.27142161181
#时尚/美妆 : -1.19213581978
#点击近一个月胸罩广告 : -2.28271002496
#女性护理 : -1262573.59031
#近一年内购买精品男包 : 11618.1687679
#爱下厨偏好极度爱好 : -1.80517707807
#美妆工具 : -7.0698178746
#近六个月搜索关键词女 : -214304327.497
#孩子3-6岁 : -2.25022980621
#半年内购买充气/仿真娃娃 : -1.40654863935
#高度关注女装用户未知 : -197.053910514
#高度关注女装用户精确 : -4.73195498224
#孕产/胎教 : -1.26783237167
#女装 : -299481.839434
#近一年内购买充气/仿真娃娃 : -1.48620079898
#男装 : 176851.792398
#半年内购买男士面部护肤 : 42569.3092312
#女装用户广泛 : -305.117706105
#母婴 : -13.4579920278
#男士面部护肤 : 338872.071481
#育儿/家教 : -1.91720023779
#抹胸 : -1.95241238575
#关注化妆品精确 : -4.73379471989
#男式内裤 : 130995.524736
#关注化妆品未知 : -7.92609168581
#商务男袜 : 72.8471245147
#婚恋与两性 : -1.23237592896
#半年内男用延时 : 115.344678106
#孩子6岁以上 : -1.36562252195
#怀孕妈妈 : -1.20663609305
#点击近一个月女性励志书籍广告 : -1.0339072794
#爱下厨偏好中度爱好 : -1.63894419656
#半年内购买避孕套 : -3.67641641217
#孩子6-12个月 : -1.70791030273
#点击近一个月仿真阳具广告 : 1
#女式内裤 : -264611.13153
#高度关注女装用户广泛 : -94.6687186687
#女装用户未知 : -1545.33080745
#珠宝首饰 : -10.8469042647
#半年内排卵验孕类目 : -1.66798386227
#孩子0-3个月 : -1.23258565166
#近一年内购买火机烟具 : -3.57293222848
#关注女装用户广泛 : -880.802098577
#关注女装用户未知 : -51207.5787415
#男士彩妆 : 3.47361529603
#美发假发/造型 : -6.03157909814
#半年内购买精品男包 : 2224.3346975
#半年内购买男士丝巾/围巾 : 9.12153934611
#近一年内购买流行男鞋 : 227333.660993
#近一年内购买时尚女鞋 : -266534.722249
#女士丝巾/围巾/披肩 : -305.406229742

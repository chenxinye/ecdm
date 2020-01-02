import numpy as np
from numpy import array
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.neural_network import MLPClassifier


class detection:

    def __init__(self,xtrain, ytrain, SEED = 2019):
        self.SEED = SEED
        models = self.get_models()
        self.P, self.model_list = Ad.train_predict(models, xtrain, ytrain)
        self.score_models(self.P,ytrain)
        
        st = len(xtrain)
        xtrain, ytrain = xtrain.reset_index(drop = True), ytrain.reset_index(drop = True)
        xtrain['label'] = ytrain
        xtrain['prediction'] = self.P.mean(axis = 1)
        xtrain['anomaly'] = 0.0
        index = xtrain[((xtrain.prediction < 0.35) & (xtrain.label == 1)) | (xtrain.prediction > 0.65) & (xtrain.label == 0)].index
        xtrain.anomaly.iloc[index] = 1
        xtrain = xtrain[xtrain.anomaly == 0]
        self.ytrain = xtrain.label
        self.xtrain = xtrain.drop('label',axis = 1)
        self.xtrain = self.xtrain.drop(['anomaly','prediction'],axis = 1)
        et = len(self.xtrain)
        print('Outlier removal ratio:{}'.format((st-et)/st))
        
    def get_models(self):
        """base learners."""
        lr = LogisticRegression(C=50, random_state=self.SEED)
        nn = MLPClassifier((80, 60), solver = 'adam', learning_rate = 'adaptive',
                           early_stopping=True, random_state=self.SEED)
        models = {'mlp-nn': nn,'logistic': lr}
        return models


    def train_predict(self, models,xtr, ytr):
        P = np.zeros((ytr.shape[0], len(models)))
        P = pd.DataFrame(P)

        print("Fitting models.")
        cols = list()
        model_list = list()
        for i, (name, m) in tqdm(enumerate(models.items())):
            print("%s..." % name, end=" ", flush=False)
            m.fit(xtr, ytr)
            P.iloc[:, i] = m.predict_proba(xtr)[:, 1]
            print("...model save...")
            model_list.append(m)
            cols.append(name)
            print("done")
        return P, model_list


    def score_models(self, P, y):
        """auc score"""
        print("Scoring models.")
        for m in tqdm(P.columns):
            score = roc_auc_score(y, P.loc[:, m])
            print("%-26s: %.3f" % (m, score))
        print("Done.\n")
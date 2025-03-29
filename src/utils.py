import os, sys
import pandas as pd
import numpy as np 
import dill
from sklearn.metrics import r2_score

from src.exceptions import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model( xtrain ,yTrain ,xTest ,yTest ,models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(xtrain, yTrain)

            y_train_pred = model.predict(xtrain)
            y_test_pred = model.predict(xTest)

            train_model_score = r2_score(yTrain, y_train_pred)
            test_model_score = r2_score(yTest, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e, sys)

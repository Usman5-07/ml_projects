import os, sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor, 
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.exceptions import CustomException 
from src.logger import logging

from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initailize_model_trainer(self, train_array, test_array):
        try:
            logging.info("Spliting training and test data")
            x_train, y_train, x_test,  y_test = (
                train_array[:,:-1], # Gets all the festures and excludes the last one
                train_array[:,-1], # Excludes all the other features and selects only the last feature from the [TRAIN] data. 
                test_array[:,:-1], # # Gets all the festures and excludes the last one from the dataset
                test_array[:,-1] # Excludes all the other features and selects only the last feature from the [TEST] data. 
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbours Regressor": KNeighborsRegressor(),
                "XGB Classifier": XGBRegressor(), 
                "Cat Boost Regressor": CatBoostRegressor(verbose=False), 
                "AdaBoostRegressor": AdaBoostRegressor()
            }
            model_report = dict=evaluate_model(
                xtrain= x_train,
                yTrain = y_train,
                xTest = x_test,
                yTest = y_test,
                models = models)
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)
                                                        ]
            best_model = models[best_model_name]

            if(best_model_score < 0.6):
                raise CustomException("No Best Model Found")
            
            logging.info("Best model is found on both training and test dataset")

            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_model
                        )
            
            predicted_score = best_model.predict(x_test)

            r2_sqrd = r2_score(y_test, predicted_score)

            return r2_sqrd, best_model_name

        except Exception as e:
            raise CustomException(e, sys)

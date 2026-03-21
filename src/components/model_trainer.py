import os
import sys

from dataclasses import dataclass

from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class modeltranerconfig:

    trained_model_file_path = os.path.join(
        "artifacts",
        "model.pkl"
    )


class modeltraner:

    def __init__(self):

        self.model_trainer_config = modeltranerconfig()

    def initiate_model_trainer(self, train_array, test_array):

        try:

            logging.info("Splitting training and testing data")

            x_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            x_test = test_array[:, :-1]
            y_test = test_array[:, -1]


            models = {

                "Random Forest": RandomForestRegressor(),

                "Decision Tree": DecisionTreeRegressor(),

                "Gradient Boosting": GradientBoostingRegressor(),

                "Linear Regression": LinearRegression(),

                "XGBRegressor": XGBRegressor(),

                "CatBoost Regressor": CatBoostRegressor(
                    verbose=False
                ),

                "AdaBoost Regressor": AdaBoostRegressor(),

            }


            model_report = evaluate_models(
                x_train,
                y_train,
                x_test,
                y_test,
                models
            )


            best_model_score = max(
                model_report.values()
            )


            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]


            best_model = models[best_model_name]


            if best_model_score < 0.6:

                raise CustomException(
                    "No best model found"
                )


            logging.info(
                "Best model found on training and testing dataset"
            )


            save_object(

                file_path=self.model_trainer_config.trained_model_file_path,

                obj=best_model
            )


            predicted = best_model.predict(
                x_test
            )


            model_r2_score = r2_score(
                y_test,
                predicted
            )


            return model_r2_score


        except Exception as e:

            raise CustomException(e, sys)
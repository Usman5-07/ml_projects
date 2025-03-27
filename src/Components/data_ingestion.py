import os, sys, pandas
from operator import index

import pandas as pd

from src.exceptions import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split

from dataclasses import dataclass

from src.Components.data_transformation import Data_Transformation

@dataclass
class DataIngestionConfig:
    train_data_path: str= os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()  # Created an object for the above class to use its variables

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")

        try:
            df = pd.read_csv(r"Notebooks\stud.csv")
            logging.info("Read the dataset as DataFrame")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train Test Split initiated...")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Saving the training Dataset to the path defined in the DataIngestionClass[Dataclass]
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            # Saving the testing Dataset to the path defined in the DataIngestionClass[Dataclass]
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of data is completed!")

            return(

                # Returning both the files paths which can will be used in data transformation component.
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise  CustomException(e, sys)



if __name__ == "__main__":
    data_ingestion_object = DataIngestion()
    train_data, test_data = data_ingestion_object.initiate_data_ingestion()

    data_transformation_obj = Data_Transformation()

    data_transformation_obj.initiate_data_transformation(train_data, test_data)

    


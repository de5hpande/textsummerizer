from src.textSummarizer.components.data_ingestion import DataIngestion
from src.textSummarizer.config.configuration import ConfigurationManager
from src.textSummarizer.logging import logger
from src.textSummarizer.entity import DataIngestionConfig

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_ingestion(self):
        config=ConfigurationManager()
        data_ingestion_config=config.get_data_ingestion_config()
        data_ingestion=DataIngestion(config=data_ingestion_config)

        data_ingestion.downlaod_file()
        data_ingestion.extract_zip_file()
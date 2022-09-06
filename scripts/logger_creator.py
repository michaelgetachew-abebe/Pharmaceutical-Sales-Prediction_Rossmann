from asyncio.log import logger
import logging

class logwritter:
    
    def __init__(self, file_name: str, basic_level = logging.INFO):

        logger = logging.getLogger(__name__)

        logger.setLevel(basic_level)

        file_handler = logging.FileHandler(file_name)

        formatter = logging.Formatter('%(asctime)s : %(levelname)s: %(name)s: %(module)s: %(funcName)s: %(message)s')

        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        self.logger = logger

    def get_logwritter(self) -> logging.Logger:
        return self.logger
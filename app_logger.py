from datetime import datetime
import logging
import os
import csv



class AppLogger:
    def __init__(self, name: str = 'cassava_logger',
                 log_folder: str = os.path.join(".", "my_logs", "app_logs")
                 ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.log_folder = log_folder
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_session_id = f'session_{current_time}'
        # Create the directory if it doesn't exist
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

        if not self.logger.handlers:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.DEBUG)
            stream_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            self.logger.addHandler(stream_handler)


    def add_file_handler_to_logger(self, file_name: str = 'cassava_logger'):
        already_exit = False
        for hdl in self.logger.handlers:
            if isinstance(hdl, logging.FileHandler):
                already_exit = True
                logger.warning('Trying to add another file handler')
                break
        if not already_exit:
            log_filename = f"{file_name}.log"
            log_file_path = os.path.join(self.log_folder, log_filename)
            file_handler = logging.FileHandler(log_file_path, mode="a")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: - %(message)s"))
            self.logger.addHandler(file_handler)
            self.logger.info("Starting new logging session with id: %s", self.log_session_id)

    def get_logger_file_path(self) -> str:
        paths = []
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                paths.append(handler.baseFilename)
        return paths[0]
    def force_log_write(self):
        for handler in self.logger.handlers:
            handler.flush()

    def info(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)
    def error(self, *args, **kwargs):
        self.logger.error(*args, **kwargs)
    def warning(self, *args, **kwargs):
        self.logger.warning(*args, **kwargs)

# will be the same object everywhere it's imported.
logger = AppLogger('cassava_logger')
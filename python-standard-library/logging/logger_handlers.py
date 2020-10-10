import logging

# return a logger with the specified name & creating it if necessary
logger = logging.getLogger(__name__)

# create a logger handler, in this case: file handler
file_handler = logging.FileHandler("file.log")
# set the level of logging to INFO
file_handler.setLevel(logging.INFO)

# create a logger formatter
logging_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# add the format to the logger handler
file_handler.setFormatter(logging_format)

# add the handler to the logger
logger.addHandler(file_handler)

# use the logger as previously
logger.critical("This is a critical message!")

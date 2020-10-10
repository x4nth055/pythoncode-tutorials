import logging

# make a basic logging configuration
# here we set the level of logging to DEBUG
logging.basicConfig(
    level=logging.DEBUG
)

# make a debug message
logging.debug("This is a simple debug log")

# make an info message
logging.info("This is a simple info log")

# make a warning message
logging.warning("This is a simple warning log")

# make an error message
logging.error("This is a simple error log")

# make a critical message
logging.critical("This is a simple critical log") 
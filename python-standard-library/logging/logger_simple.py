import logging

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

# just mapping logging level integers into strings for convenience
logging_levels = {
    logging.DEBUG: "DEBUG", # 10
    logging.INFO: "INFO", # 20
    logging.WARNING: "WARNING", # 30
    logging.ERROR: "ERROR", # 40
    logging.CRITICAL: "CRITICAL", # 50
}

# get the current logging level
print("Current logging level:", logging_levels.get(logging.root.level))

# get the current logging format
print("Current logging format:", logging.BASIC_FORMAT)
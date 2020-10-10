import logging
import math

logging.basicConfig(level=logging.DEBUG,
                    handlers=[logging.FileHandler('logs.log', 'a', 'utf-8')],
                    format="%(asctime)s %(levelname)-6s - %(funcName)-8s - %(filename)s - %(lineno)-3d - %(message)s",
                    datefmt="[%Y-%m-%d] %H:%M:%S - ",
                    )

logging.info("This is an info log")

def square_root(x):
    logging.debug(f"Getting the square root of {x}") 
    try:
        result = math.sqrt(x)
    except ValueError:
        logging.exception("Cannot get square root of a negative number")
        # or
        # logging.error("Cannot get square root of a negative number", exc_info=True)
        return None
    logging.info(f"The square root of {x} is {result:.5f}")
    return result

square_root(5)
square_root(-5)
import datetime
from pygtail import Pygtail

# Loading the package called re from the RegEx Module in order to work with Regular Expressions
import re


class FileChecker:
    def __init__(self, exceptionPattern):
        self.exceptionPattern = exceptionPattern

    def checkForException(self, event, path):
        # Get current date and time according to the specified format.
        now = (datetime.datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
        # Read the lines of the file (specified in the path) that have not been read yet
        # Meaning by that it will start from the point where it was last stopped.
        for num, line in enumerate(Pygtail(path), 1):
            # Remove leading and trailing whitespaces including newlines.
            line = line.strip()
            # Return all non-overlapping matches of the values specified in the Exception Pattern.
            # The line is scanned from left to right and matches are returned in the oder found.
            if line and any(re.findall('|'.join(self.exceptionPattern), line, flags=re.I | re.X)):
                # Observation Detected
                type = 'observation'
                msg = f"{now} -- {event.event_type} -- File: {path} -- Observation: {line}"
                yield type, msg
            elif line:
                # No Observation Detected
                type = 'msg'
                msg = f"{now} -- {event.event_type} -- File: {path}"
                yield type, msg

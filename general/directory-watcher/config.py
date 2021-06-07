# Application configuration File
################################

# Directory To Watch, If not specified, the following value will be considered explicitly.
WATCH_DIRECTORY = "C:\\SCRIPTS"

# Delay Between Watch Cycles In Seconds
WATCH_DELAY = 1

# Check The WATCH_DIRECTORY and its children
WATCH_RECURSIVELY = False

# whether to watch for directory events
DO_WATCH_DIRECTORIES = True

# Patterns of the files to watch
WATCH_PATTERN = '.txt,.trc,.log'

LOG_FILES_EXTENSIONS = ('.txt', '.log', '.trc')

# Patterns for observations
EXCEPTION_PATTERN = ['EXCEPTION', 'FATAL', 'ERROR']

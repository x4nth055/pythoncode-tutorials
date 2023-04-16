import argparse

class CustomAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # Perform custom processing on the argument values
        processed_values = [value.upper() for value in values]

        # Set the attribute on the namespace object
        setattr(namespace, self.dest, processed_values)

# Set up argument parser and add the custom action
parser = argparse.ArgumentParser(description='Custom argument action example.')
parser.add_argument('-n', '--names', nargs='+', action=CustomAction, help='A list of names to be processed.')

args = parser.parse_args()
print(args.names)

import argparse

parser = argparse.ArgumentParser(description='A simple argparse example.')
parser.add_argument('--foo', action='store', help='Store the value of foo.')
parser.add_argument('--enable', action='store_true', help='Enable the feature.')
parser.add_argument('--disable', action='store_false', help='Disable the feature.')
parser.add_argument('--level', action='store_const', const='advanced', help='Set level to advanced.')
parser.add_argument('--values', action='append', help='Append values to a list.')
parser.add_argument('--add_const', action='append_const', const=42, help='Add 42 to the list.')
parser.add_argument('-v', '--verbose', action='count', help='Increase verbosity level.')
args = parser.parse_args()
print(f"Values: {args.values}")
print(f"Verbosity: {args.verbose}")

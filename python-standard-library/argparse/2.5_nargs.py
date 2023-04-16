import argparse

parser = argparse.ArgumentParser(description='A simple argparse example.')
parser.add_argument('--values', nargs=3)
# parser.add_argument('--value', nargs='?', default='default_value')
# parser.add_argument('--values', nargs='*')
# parser.add_argument('--values', nargs='+')

args = parser.parse_args()
print(f"Values: {args.values}")

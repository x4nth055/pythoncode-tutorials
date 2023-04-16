import argparse

parser = argparse.ArgumentParser(description='A simple argparse example.')
parser.add_argument('input', help='Input file to process.')

args = parser.parse_args()
print(f'Processing file: {args.input}')

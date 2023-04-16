import argparse

parser = argparse.ArgumentParser(description='A simple argparse example.')
parser.add_argument('input', help='Input file to process.')
parser.add_argument('-m', '--mode', choices=['add', 'subtract', 'multiply', 'divide'], help='Calculation mode.')

args = parser.parse_args()
print(f'Processing file: {args.input}')
print(f"Mode: {args.mode}")

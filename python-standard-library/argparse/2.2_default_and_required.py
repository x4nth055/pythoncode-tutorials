import argparse

parser = argparse.ArgumentParser(description='A simple argparse example.')
parser.add_argument('input', help='Input file to process.')
# parser.add_argument('-o', '--output', default='output.txt', help='Output file.')
parser.add_argument('-o', '--output', required=True, help='Output file.')

args = parser.parse_args()
print(f'Processing file: {args.input}')
print(f"Writing to file: {args.output}")

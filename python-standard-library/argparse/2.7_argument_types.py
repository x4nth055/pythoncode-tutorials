import argparse

parser = argparse.ArgumentParser(description='A simple argparse example.')
parser.add_argument("-r", "--ratio", type=float)
args = parser.parse_args()
print(f"Ratio: {args.ratio}")

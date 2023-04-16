import argparse

# Operation functions
def add(args):
    print(args.x + args.y)

def subtract(args):
    print(args.x - args.y)

def multiply(args):
    print(args.x * args.y)

def divide(args):
    print(args.x / args.y)

# Set up argument parser
parser = argparse.ArgumentParser(description='Command-line calculator.')
subparsers = parser.add_subparsers()

# Add subcommands
add_parser = subparsers.add_parser('add', help='Add two numbers.')
add_parser.add_argument('x', type=float, help='First number.')
add_parser.add_argument('y', type=float, help='Second number.')
add_parser.set_defaults(func=add)

subtract_parser = subparsers.add_parser('subtract', help='Subtract two numbers.')
subtract_parser.add_argument('x', type=float, help='First number.')
subtract_parser.add_argument('y', type=float, help='Second number.')
subtract_parser.set_defaults(func=subtract)

multiply_parser = subparsers.add_parser('multiply', help='Multiply two numbers.')
multiply_parser.add_argument('x', type=float, help='First number.')
multiply_parser.add_argument('y', type=float, help='Second number.')
multiply_parser.set_defaults(func=multiply)

divide_parser = subparsers.add_parser('divide', help='Divide two numbers.')
divide_parser.add_argument('x', type=float, help='First number.')
divide_parser.add_argument('y', type=float, help='Second number.')
divide_parser.set_defaults(func=divide)

args = parser.parse_args()
args.func(args)

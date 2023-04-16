import argparse
import os

# Rename function
def rename_files(args):
    # Your file renaming logic here
    print(f"Renaming files in {args.path}...")
    print(f"Prefix: {args.prefix}")
    print(f"Suffix: {args.suffix}")
    print(f"Replace: {args.replace}")
    os.chdir(args.path)
    for file in os.listdir():
        # Get the file name and extension
        file_name, file_ext = os.path.splitext(file)
        # Add prefix
        if args.prefix:
            file_name = f"{args.prefix}{file_name}"
        # Add suffix
        if args.suffix:
            file_name = f"{file_name}{args.suffix}"
        # Replace substring
        if args.replace:
            file_name = file_name.replace(args.replace[0], args.replace[1])
        # Rename the file
        print(f"Renaming {file} to {file_name}{file_ext}")
        os.rename(file, f"{file_name}{file_ext}")
        
# custom type for checking if a path exists
def path_exists(path):
    if os.path.exists(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"Path {path} does not exist.")
    
    
# Set up argument parser
parser = argparse.ArgumentParser(description='File renaming tool.')
parser.add_argument('path', type=path_exists, help='Path to the folder containing the files to rename.')
parser.add_argument('-p', '--prefix', help='Add a prefix to each file name.')
parser.add_argument('-s', '--suffix', help='Add a suffix to each file name.')
parser.add_argument('-r', '--replace', nargs=2, help='Replace a substring in each file name. Usage: -r old_string new_string')

args = parser.parse_args()

# Call the renaming function
rename_files(args)

# Import re for regex functions
import re

# Import sys for getting the command line arguments
import sys

# Import docx to work with .docx files.
# Must be installed: pip install python-docx
from docx import Document

# Check if Command Line Arguments are passed.
if len(sys.argv) < 3:
    print('Not Enough arguments where supplied')
    sys.exit()

# Check if replacers are in a valid schema
for replaceArg in sys.argv[2:]:
    if len(replaceArg.split('=')) != 2:
        print('Faulty replace argument given')
        print('-> ', replaceArg)
        sys.exit()

# Store file path from CL Arguments.
file_path = sys.argv[1]

if file_path.endswith('.docx'):
    doc = Document(file_path)
    # Loop through replacer arguments
    occurences = {}
    for replaceArgs in sys.argv[2:]:
        # split the word=replacedword into a list
        replaceArg = replaceArgs.split('=')
        # initialize the number of occurences of this word to 0
        occurences[replaceArg[0]] = 0
        # Loop through paragraphs
        for para in doc.paragraphs:
            # Loop through runs (style spans)
            for run in para.runs:
                # if there is text on this run, replace it
                if run.text:
                    # get the replacement text
                    replaced_text = re.sub(replaceArg[0], replaceArg[1], run.text, 999)
                    if replaced_text != run.text:
                        # if the replaced text is not the same as the original
                        # replace the text and increment the number of occurences
                        run.text = replaced_text
                        occurences[replaceArg[0]] += 1
                    
    # print the number of occurences of each word
    for word, count in occurences.items():
        print(f"The word {word} was found and replaced {count} times.")
    
    # make a new file name by adding "_new" to the original file name
    new_file_path = file_path.replace(".docx", "_new.docx")
    # save the new docx file
    doc.save(new_file_path)
else:
    print('The file type is invalid, only .docx are supported')

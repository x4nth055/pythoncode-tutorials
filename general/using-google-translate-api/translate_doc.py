from googletrans import Translator
import argparse
import os

# init the translator
translator = Translator()

def translate(text, src="auto", dest="en"):
    """Translate `text` from `src` language to `dest`"""
    return translator.translate(text, src=src, dest=dest).text
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Python script to translate text using Google Translate API (googletrans wrapper)")
    parser.add_argument("target", help="Text/Document to translate")
    parser.add_argument("-s", "--source", help="Source language, default is Google Translate's auto detection", default="auto")
    parser.add_argument("-d", "--destination", help="Destination language, default is English", default="en")
    
    args = parser.parse_args()
    target = args.target
    src = args.source
    dest = args.destination
    
    if os.path.isfile(target):
        # translate a document instead
        # get basename of file
        basename = os.path.basename(target)
        # get the path dir
        dirname = os.path.dirname(target)
        try:
            filename, ext = basename.split(".")
        except:
            # no extension
            filename = basename
            ext = ""

        translated_text = translate(open(target).read(), src=src, dest=dest)
        # write to new document file
        open(os.path.join(dirname, f"{filename}_{dest}{f'.{ext}' if ext else ''}"), "w").write(translated_text)
    else:
        # not a file, just text, print the translated text to standard output
        print(translate(target, src=src, dest=dest))
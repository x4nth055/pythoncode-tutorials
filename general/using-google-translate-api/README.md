# [How to Translate Text in Python](https://www.thepythoncode.com/article/translate-text-in-python)
To run this:
- `pip3 install -r requirements.txt`
- Tutorial code is in `translator.py`.
- If you want to translate a document, you can use `translate_doc.py`:
    ```
    python3 translate_doc.py --help
    ```
    **Output:**
    ```
    usage: translate_doc.py [-h] [-s SOURCE] [-d DESTINATION] target

    Simple Python script to translate text using Google Translate API (googletrans
    wrapper)

    positional arguments:
    target                Text/Document to translate

    optional arguments:
    -h, --help            show this help message and exit
    -s SOURCE, --source SOURCE
                            Source language, default is Google Translate's auto
                            detection
    -d DESTINATION, --destination DESTINATION
                            Destination language, default is English
    ```
- For instance, if you want to translate text in the document `wonderland.txt` from english (`en` as language code) to arabic (`ar` as language code):
    ```
    python translate_doc.py wonderland.txt --source en --destination ar
    ```
    A new file `wonderland_ar.txt` will appear in the current directory that contains the translated document.
- You can also translate text and print in the stdout using `translate_doc.py`:
    ```
    python translate_doc.py 'Bonjour' -s fr -d en
    ```
    **Output:**
    ```
    'Hello'
    ```

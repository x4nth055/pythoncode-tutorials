from googletrans import Translator, constants
from pprint import pprint

# init the Google API translator
translator = Translator()

# translate a spanish text to english text (by default)
translation = translator.translate("Hola Mundo")
print(f"{translation.origin} ({translation.src}) --> {translation.text} ({translation.dest})")

# translate a spanish text to arabic for instance
translation = translator.translate("Hola Mundo", dest="ar")
print(f"{translation.origin} ({translation.src}) --> {translation.text} ({translation.dest})")

# specify source language
translation = translator.translate("Wie gehts ?", src="de")
print(f"{translation.origin} ({translation.src}) --> {translation.text} ({translation.dest})")
# print all translations and other data
pprint(translation.extra_data)

# translate more than a phrase
sentences = [
    "Hello everyone",
    "How are you ?",
    "Do you speak english ?",
    "Good bye!"
]
translations = translator.translate(sentences, dest="tr")
for translation in translations:
    print(f"{translation.origin} ({translation.src}) --> {translation.text} ({translation.dest})")

# detect a language
detection = translator.detect("नमस्ते दुनिया")
print("Language code:", detection.lang)
print("Confidence:", detection.confidence)
# print the detected language
print("Language:", constants.LANGUAGES[detection.lang])

# print all available languages
print("Total supported languages:", len(constants.LANGUAGES))
print("Languages:")
pprint(constants.LANGUAGES)
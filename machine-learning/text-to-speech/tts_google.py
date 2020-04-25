import gtts
from playsound import playsound

# make request to google to get synthesis
tts = gtts.gTTS("Hello world")
# save the audio file
tts.save("hello.mp3")
# play the audio file
playsound("hello.mp3")

# in spanish
tts = gtts.gTTS("Hola Mundo", lang="es")
tts.save("hola.mp3")
playsound("hola.mp3")

# all available languages along with their IETF tag
print(gtts.lang.tts_langs())
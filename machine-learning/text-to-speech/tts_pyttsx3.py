import pyttsx3

# initialize Text-to-speech engine
engine = pyttsx3.init()

# convert this text to speech
text = "Python is a great programming language"
engine.say(text)
# play the speech
engine.runAndWait()

# get details of speaking rate
rate = engine.getProperty("rate")
print(rate)

# setting new voice rate (faster)
engine.setProperty("rate", 300)
engine.say(text)
engine.runAndWait()

# slower
engine.setProperty("rate", 100)
engine.say(text)
engine.runAndWait()

# get details of all voices available
voices = engine.getProperty("voices")
print(voices)
# set another voice
engine.setProperty("voice", voices[1].id)
engine.say(text)
engine.runAndWait()

# saving speech audio into a file
engine.save_to_file(text, "python.mp3")
engine.runAndWait()
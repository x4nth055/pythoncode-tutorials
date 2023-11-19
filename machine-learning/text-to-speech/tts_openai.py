from openai import OpenAI

# initialize the OpenAI API client
api_key = "YOUR_OPENAI_API_KEY"
client = OpenAI(api_key=api_key)

# sample text to generate speech from
text = """In his miracle year, he published four groundbreaking papers. 
These outlined the theory of the photoelectric effect, explained Brownian motion, 
introduced special relativity, and demonstrated mass-energy equivalence."""

# generate speech from the text
response = client.audio.speech.create(
    model="tts-1", # the model to use, there is tts-1 and tts-1-hd
    voice="nova", # the voice to use, there is alloy, echo, fable, onyx, nova, and shimmer
    input=text, # the text to generate speech from
    speed=1.0, # the speed of the generated speech, ranging from 0.25 to 4.0
)
# save the generated speech to a file
response.stream_to_file("openai-output.mp3")
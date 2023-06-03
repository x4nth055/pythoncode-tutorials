import evaluate

wer = evaluate.load("wer")

# reference = "the cat sat on the mat"
# hypothesis = "the cat mat"
reference = "The cat is sleeping on the mat."
hypothesis = "The cat is playing on mat."
print(wer.compute(references=[reference], predictions=[hypothesis]))
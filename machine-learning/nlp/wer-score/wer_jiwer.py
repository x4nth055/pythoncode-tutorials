from jiwer import wer

if __name__ == "__main__":
    # reference = "the cat sat on the mat"
    # hypothesis = "the cat mat"
    reference = "The cat is sleeping on the mat."
    hypothesis = "The cat is playing on mat."
    print(wer(reference, hypothesis))
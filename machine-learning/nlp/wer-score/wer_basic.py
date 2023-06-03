def calculate_wer(reference, hypothesis):
	ref_words = reference.split()
	hyp_words = hypothesis.split()
 
	# Counting the number of substitutions, deletions, and insertions
	substitutions = sum(1 for ref, hyp in zip(ref_words, hyp_words) if ref != hyp)
	deletions = len(ref_words) - len(hyp_words)
	insertions = len(hyp_words) - len(ref_words)
 
	# Total number of words in the reference text
	total_words = len(ref_words)
 
	# Calculating the Word Error Rate (WER)
	wer = (substitutions + deletions + insertions) / total_words
	return wer


if __name__ == "__main__":
    reference = "the cat sat on the mat"
    hypothesis = "the cat mat"
    print(calculate_wer(reference, hypothesis))
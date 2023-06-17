from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Single reference
candidate_summary = "the cat was found under the bed"
reference_summary = "the cat was under the bed"
scores = scorer.score(reference_summary, candidate_summary)
for key in scores:
   print(f'{key}: {scores[key]}')

# Multiple references
candidate_summary = "the cat was found under the bed"
reference_summaries = ["the cat was under the bed", "found a cat under the bed"]
scores = {key: [] for key in ['rouge1', 'rouge2', 'rougeL']}
for ref in reference_summaries:
   temp_scores = scorer.score(ref, candidate_summary)
   for key in temp_scores:
       scores[key].append(temp_scores[key])

for key in scores:
   print(f'{key}:\n{scores[key]}')
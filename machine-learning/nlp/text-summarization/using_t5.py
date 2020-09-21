from transformers import T5ForConditionalGeneration, T5Tokenizer

# initialize the model architecture and weights
model = T5ForConditionalGeneration.from_pretrained("t5-base")
# initialize the model tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")
article = """
Justin Timberlake and Jessica Biel, welcome to parenthood. 
The celebrity couple announced the arrival of their son, Silas Randall Timberlake, in statements to People. 
"Silas was the middle name of Timberlake's maternal grandfather Bill Bomar, who died in 2012, while Randall is the musician's own middle name, as well as his father's first," People reports. 
The couple announced the pregnancy in January, with an Instagram post. It is the first baby for both.
"""
# encode the text into tensor of integers using the appropriate tokenizer
inputs = tokenizer.encode("summarize: " + article, return_tensors="pt", max_length=512, truncation=True)
# generate the summarization output
outputs = model.generate(
    inputs, 
    max_length=150, 
    min_length=40, 
    length_penalty=2.0, 
    num_beams=4, 
    early_stopping=True)
# just for debugging
print(outputs)
print(tokenizer.decode(outputs[0]))
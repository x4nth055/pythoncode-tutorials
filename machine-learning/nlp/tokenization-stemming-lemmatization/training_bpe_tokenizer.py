from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

tk = Tokenizer(BPE(unk_token="[UNK]"))
tr = BpeTrainer()
tk.pre_tokenizer = Whitespace()

f = [f"wikitext-103-raw\wiki.{s}.raw" for s in ["test", "train", "valid"]]
tk.train(f, tr)

tk.save("tokenizer-wiki.json")

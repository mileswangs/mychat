from mychat.tokenizer import get_tokenizer

tokenizer = get_tokenizer()
assistant_end = tokenizer.encode_special("<|assistant_end|>")
bos = tokenizer.encode_special("<|bos|>")
vocab_size = tokenizer.get_vocab_size()

print("bos Token:", bos)  # Example output: [50256]
print("Assistant End Token ID:", assistant_end)  # Example output: [50256]
print("Vocabulary Size:", vocab_size)  # Example output: 50257

import torch
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("jcblaise/roberta-tagalog-base")
model = AutoModel.from_pretrained("jcblaise/roberta-tagalog-base")
model.eval()

sentence = "Bukas ay magbubukas ang tindahan."

encoded = tokenizer(sentence, return_tensors="pt", return_offsets_mapping=True)
word_ids = encoded.word_ids()
offsets = encoded["offset_mapping"][0].tolist()
tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])

with torch.no_grad():
    out = model(**{k: v for k, v in encoded.items() if k != "offset_mapping"})
    hidden = out.last_hidden_state[0]  # (R, 768)

for i, (token, wid, offset, vec) in enumerate(zip(tokens, word_ids, offsets, hidden)):
    print(
        f"idx={i:3d} | token={token:15s} | word_id={str(wid):4s} | offset={str(offset):12s} | vec[:4]={vec[:4].tolist()}"
    )

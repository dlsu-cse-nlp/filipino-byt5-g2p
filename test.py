from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("charsiu/g2p_multilingual_byT5_small_100")
print(tok.convert_tokens_to_ids("'"))

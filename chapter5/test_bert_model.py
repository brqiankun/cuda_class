import torch
from transformers import BertTokenizer, BertModel

# BERT_PATH = "/home/rui.bai/bairui_file/cuda_learning/cuda_class/chapter5/bert-base-uncased"
BERT_PATH = "/home/br/program/bert_origin"
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
model = BertModel.from_pretrained(BERT_PATH)
text = "hello [MASK]"
print(tokenizer.tokenize(text))
encoded_input = tokenizer(text, return_tensors = 'pt')
output = model(**encoded_input)
print(output)


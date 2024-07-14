from transformers import BertModel, BertTokenizer
import torch

model_name = 'bert-base-uncased'
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)
model.eval()

with open("bert_vocab.txt", "w+", encoding="utf-8") as file:
    vocab = tokenizer.get_vocab()
    for token, index in vocab.items():
        file.write(f'{token} {index}\n')

example_inputs = tokenizer("Example input. I dont exactly know why this is needed. Apparently for 'shape inference'", return_tensors="pt")

print(example_inputs)
print(example_inputs["input_ids"])
print(example_inputs["attention_mask"])

torch.onnx.export(model,
                  args=(example_inputs['input_ids'], example_inputs['attention_mask']),
                  f="bert-base-uncased.onnx",
                  export_params=True,
                  input_names=['input_ids', 'attention_mask'],
                  output_names=['output'],
                  dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                                'output': {0: 'batch_size', 1: 'sequence_length'}},
                    opset_version=15)


import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("bert-base-uncased.onnx")

inputs = tokenizer("Your new input text here", return_tensors='np', truncation=True, max_length=512, padding="max_length")
print("INPUTSSS")
print(inputs["input_ids"], inputs["attention_mask"])
onnx_inputs = {session.get_inputs()[0].name: np.array(inputs['input_ids'], dtype=np.int64), session.get_inputs()[1].name: np.array(inputs['attention_mask'], dtype=np.int64)}

onnx_outputs = session.run(None, onnx_inputs)
    
embeddings = np.mean(onnx_outputs[0], axis=1)
print(embeddings)
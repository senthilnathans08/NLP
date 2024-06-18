from transformers import BertModel, AutoTokenizer
from scipy.spatial.distance import cosine

model_name ='bert-base-cased'
model = BertModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

text="Tokenize me this please"
def predict(text):
    encoded_inputs=tokenizer(text,return_tensors='pt')
    return model(**encoded_inputs)[0]

sentence1 = "There was a fly drinking from my soup"
sentence2 = "To become a commerical pilot,he had to fly for 1500 hours"

tokenize1=tokenizer.tokenize(sentence1)
tokenize2=tokenizer.tokenize(sentence2)


out1 = predict(sentence1)
out2 = predict(sentence2)

emb1= out1[0:, tokenize1.index("fly"), :].detach()
emb2= out2[0:, tokenize2.index("fly"), :].detach()

emb1.flatten()
emb2.shape

cosine(emb1.flatten(), emb2.flatten())
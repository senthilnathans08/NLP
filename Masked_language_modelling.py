from transformers import AutoTokenizer, AutoModelForMaskedLM
from scipy.special import softmax
import numpy as np
model_name = "bert-base-cased"


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

mask = tokenizer.mask_token

sentence = f'I want to {mask} pizza for tonight.'

tokens= tokenizer.tokenize(sentence)

encoded_inputs= tokenizer(sentence,return_tensors='pt')

outputs = model(**encoded_inputs)
outputs
logits = outputs.logits.detach().numpy()[0]
logits.shape

mask_logits = logits[tokens.index(mask)+1]
mask_logits
Confidence_score =softmax(mask_logits)

Confidence_score.shape


for i in np.argsort(Confidence_score)[::-1][:5]:
    predict_token = tokenizer.decode(i)
    score = Confidence_score[i]
    #print(predict_token,score)
    print(sentence.replace(mask,predict_token),score)
    

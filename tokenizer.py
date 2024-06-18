from transformers import AutoTokenizer, BertModel
import pandas as pd

model_name = "bert-base-cased"


model = BertModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
sentence = AutoTokenizer.from_pretrained(model_name)

vocab=tokenizer.vocab

sentence="Conda is an open-source, cross-platform, language-agnostic package manager and environment management system."

token=tokenizer.tokenize(sentence)
token
vocab

token_ids=tokenizer.encode(sentence)


vocab_df = pd.DataFrame({"token":vocab.keys(),"token_ids":vocab.values()})

vocab_df = vocab_df.sort_values(by='token_ids').set_index('token_ids')


 

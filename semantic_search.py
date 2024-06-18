from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
import torch

dataset = load_dataset("multi_news",split="test",trust_remote_code=True)
dataset
df=dataset.to_pandas().sample(2000,random_state=42)
model = SentenceTransformer("all-MiniLM-L6-v2")


passage_embeddings=list(model.encode(df['summary'].to_list()))

passage_embeddings[0].shape

query="find me some articles about technology and artificial intelligence"

def find_revelant_news(query:str):
    
    query_embedding = model.encode(query)
    similarities= util.cos_sim(query_embedding,passage_embeddings)
    top_indices= torch.topk(similarities.flatten(),k=3).indices
    top_relevant_passages = [df.iloc[x.item()]['summary'][:200] + "..." for x in top_indices]

    return top_relevant_passages
find_revelant_news("stock market")
find_revelant_news("Natural disater")
find_revelant_news("sports")

import lmppl
import torch
import pandas as pd

from tqdm.auto import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tqdm.pandas()

df = pd.read_csv("./data/wow-gb-dataset/template_dataset.csv")

model_name = "meta-llama/Llama-3.2-1B"

scorer = lmppl.LM(model_name)

def get_perplexity(template):
    results = scorer.get_perplexity(template)
    return results

df['perplexity'] = df['text'].progress_apply(get_perplexity)

modified_name = (model_name).replace("/", "_")
df.to_csv(f'./results/1_{modified_name}_results.csv', index=False)

print(df.head())

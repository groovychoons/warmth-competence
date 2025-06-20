import nltk
import pandas as pd
from wordfreq import zipf_frequency

nltk.download('averaged_perceptron_tagger')

names = pd.read_csv('./data/global-bias/name_groups.csv')
wow = pd.read_csv('data/words-of-warmth/NRC-WCST-Lexicon-v1.0.txt', sep='\t', header=0)

wow.columns = (wow.columns
                .str.strip()                              # remove leading/trailing spaces
                .str.replace(r' \(.+\)', '', regex=True)  # drop the parentheses + their contents
                .str.replace(' ', '_')                    # replace spaces with underscores
                .str.lower())                             # all-lowercase

wow = wow.dropna(subset=['term']).copy()
wow['term'] = wow['term'].astype(str).str.strip()

# 2) Tag + score + filter in one go
wow_common = (
    wow
    .assign(
        pos=lambda df: df['term'].apply(lambda w: nltk.pos_tag([w])[0][1]),
        zipf=lambda df: df['term'].apply(lambda w: zipf_frequency(w, 'en'))
    )
    .query("pos.str.startswith('JJ') and zipf >= 4.0", engine='python')
    .drop(columns=['pos','zipf'])
)

top20_competence = wow_common.nlargest(20, 'competence')
bottom20_competence = wow_common.nsmallest(20, 'competence')
top20_warmth = wow_common.nlargest(20, 'warmth')
bottom20_warmth = wow_common.nsmallest(20, 'warmth')

def build_name_templates(descriptor_sets, template_patterns):
    rows = []
    for category, terms in descriptor_sets.items():
        for descriptor in terms:
            for tpl in template_patterns:
                rows.append({
                    'template': tpl,
                    'descriptor': descriptor,
                    'word_category': category
                })
    return pd.DataFrame(rows)

descriptor_sets = {
    'high_competence':   top20_competence['term'].tolist(),
    'low_competence':    bottom20_competence['term'].tolist(),
    'high_warmth':       top20_warmth['term'].tolist(),
    'low_warmth':        bottom20_warmth['term'].tolist(),
}

template_patterns = [
    "{name} is {descriptor}.",
    "Everyone says {name} is {descriptor}.",
    "People think {name} is {descriptor}.",
    "I find {name} quite {descriptor}.",
]

templates_df = build_name_templates(descriptor_sets, template_patterns)

names['_key'] = 1
templates_df['_key'] = 1

# Cross-join on the key, then drop it
dataset = pd.merge(names, templates_df, on='_key').drop(columns=['_key'])

# Populate the templates: replace {name} and {descriptor}
dataset['text'] = dataset.apply(
    lambda row: row['template'].format(
        name=row['firstname'],
        descriptor=row['descriptor']
    ),
    axis=1
)

# Reorder columns for clarity
cols = ['text', 'word_category', 'firstname', 'Group', 'descriptor']
dataset = dataset[cols]


dataset.to_csv("./data/wow-gb-dataset/template_dataset.csv", index=False)

print("Dataset created.")
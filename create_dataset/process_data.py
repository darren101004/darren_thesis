import pandas as pd

input_file = "data/enhanced_pipeline_gemini.csv"
df = pd.read_csv(input_file)

def get_text(row):
    rewrite = str(row['rewrite']) if not pd.isna(row['rewrite']) else ''
    if rewrite.strip() == '' or rewrite.lower().strip() == 'nan':
        return row['original_text']
    else:
        return row['rewrite']

df['rewritten_text'] = df.apply(get_text, axis=1)


output_df = df[['id', 'original_text', 'rewritten_text', 'category']]

output_df.to_csv('data/nsfw_prompts.csv', index=False, header=True)
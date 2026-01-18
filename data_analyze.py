import fastparquet as fp
from tabulate import tabulate

pf = fp.ParquetFile('data/Output_data.parquet')
df = pf.to_pandas()

print('\ntotal data amount:', len(df), '\n')

counter = {}
difficulty = {}

for index, row in df.iterrows():
    language = row['programming_language']
    diff = row['adjective']
    counter[language] = counter.get(language, 0) + 1
    difficulty[diff] = difficulty.get(diff, 0) + 1


table_data = [[language, count] for language, count in counter.items()]
difficulty_data = [[diff, count] for language, count in difficulty.items()]
headers = ["programming_language", "amount"]
difficulty_headers = ["difficulty", "amount"]

print(tabulate(table_data, headers=headers, tablefmt="grid"))
print(tabulate(difficulty_data, headers=difficulty_headers, tablefmt="grid"))

print("output random data:")
row = df.sample(1)
print('language: ', row['programming_language'].iloc[0],
      'adjective: ', row['adjective'].iloc[0],
      '\n', row['response'].iloc[0])



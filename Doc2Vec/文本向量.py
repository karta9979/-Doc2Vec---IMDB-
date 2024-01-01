import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from tqdm import tqdm 

csv_file_path = r"C:\Users\User\OneDrive\桌面\Doc2Vec\IMDB.csv"

# 嘗試使用不同的編碼讀取 CSV 文件(我這邊遇到UnicodeDecodeError所以用不同的嘗試)
encodings = ['utf-8', 'ISO-8859-1', 'cp1252']

for encoding in encodings:
    try:
        data = pd.read_csv(csv_file_path, encoding=encoding)
        print(f"Successfully read the CSV file with encoding: {encoding}")
        break
    except UnicodeDecodeError:
        print(f"Failed to read the CSV file with encoding: {encoding}")

# 保留 'review', 'sentiment' 的部分
data = data[['review', 'sentiment']]

# 使用 TaggedDocument 將每個文本與標籤相關聯
tagged_data = [TaggedDocument(words=word_tokenize(text.lower()), tags=[str(i)]) for i, text in enumerate(data['review'])]

# 訓練 Doc2Vec 模型
model = Doc2Vec(vector_size=20, window=2, min_count=1, workers=4, epochs=100)
model.build_vocab(tagged_data)

# 使用 tqdm 包裝迴圈以顯示進度條
with tqdm(total=model.epochs, desc="Training Doc2Vec") as pbar:
    for epoch in range(model.epochs):
        model.train(tagged_data, total_examples=model.corpus_count, epochs=1)
        pbar.update(1)  # 更新進度條

# 獲取整個資料集文本的向量表示
vectorized_texts = [model.dv[str(i)] for i in range(len(data))]

# 將向量列表轉換為 DataFrame
vector_df = pd.DataFrame(vectorized_texts, columns=[f'vec_{i}' for i in range(20)])

# 合併向量和原始數據
result_df = pd.concat([data, vector_df], axis=1)

# 保存結果到 CSV 文件
result_df.to_csv('output.csv', index=False)


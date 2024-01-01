import pandas as pd
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from gensim.parsing.preprocessing import preprocess_string, remove_stopwords


csv_file_path = r"C:\Users\User\OneDrive\桌面\Doc2Vec\output.csv"

# 嘗試使用不同的編碼讀取CSV文件
encodings = ['utf-8', 'ISO-8859-1', 'cp1252']

for encoding in encodings:
    try:
        data = pd.read_csv(csv_file_path, encoding=encoding)
        print(f"Successfully read the CSV file with encoding: {encoding}")
        break
    except UnicodeDecodeError:
        print(f"Failed to read the CSV file with encoding: {encoding}")

#print(type(data))
#print(data) #看資料
#print(data.describe()) #看統計資訊
#print(data.info()) 



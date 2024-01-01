import pandas as pd
import matplotlib.pyplot as plt

# 嘗試使用不同的編碼讀取CSV文件
csv_file_path = r"C:\Users\User\OneDrive\桌面\Doc2Vec\output.csv"
encodings = ['utf-8', 'ISO-8859-1', 'cp1252']

for encoding in encodings:
    try:
        data = pd.read_csv(csv_file_path, encoding=encoding)
        print(f"Successfully read the CSV file with encoding: {encoding}")
        break
    except UnicodeDecodeError:
        print(f"Failed to read the CSV file with encoding: {encoding}")

# 保留 'vec_0' 到 'vec_19' 的部分
data = data[['vec_0', 'vec_1', 'vec_2', 'vec_3', 'vec_4', 
             'vec_5', 'vec_6', 'vec_7', 'vec_8', 'vec_9', 
             'vec_10', 'vec_11', 'vec_12', 'vec_13', 'vec_14', 
             'vec_15', 'vec_16', 'vec_17', 'vec_18', 'vec_19']]


example_row = data.iloc[0]  # 假設用第一列的數據
vector_example = example_row.values

# 定義 show_image 函數
def show_image(vector):
    fig, ax = plt.subplots(1, 1, figsize=(10, 2))
    ax.tick_params(axis='both',
                   which='both',
                   left=False,
                   bottom=False,
                   top=False,
                   labelleft=False,
                   labelbottom=False)
    ax.grid(False)
    ax.bar(range(len(vector)), vector, 0.5)

show_image(vector_example)
plt.show()


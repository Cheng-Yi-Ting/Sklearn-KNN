from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import train_test_split
# 導入文本特徵矢量轉化模塊
from sklearn.feature_extraction.text import CountVectorizer
# 導入KNN
from sklearn import neighbors
# from sklearn.neighbors import NearestNeighbors
# 模型評估模塊
from sklearn.metrics import classification_report

'''
KNN
'''

'''
1 讀取數據部分
'''
# 該api會即使聯網下載數據
news = fetch_20newsgroups(subset="all")
# 檢查數據規模和細節
# print(len(news.data))
# print(news.data[0])

'''
2 分割數據部分
'''
x_train, x_test, y_train, y_test = train_test_split(news.data,
                                                    news.target,
                                                    test_size=0.25,
                                                    random_state=33)


'''
3 KNN對新聞進行預測
'''
# 進行文本轉化為特徵
vec = CountVectorizer()
x_train = vec.fit_transform(x_train)
x_test = vec.transform(x_test)
# 初始化KNN模型
knn = neighbors.KNeighborsClassifier(n_neighbors=3)
# 訓練集合上進行訓練， 估計參數
knn.fit(x_train, y_train)
# 對測試集合進行預測 保存預測結果
y_predict = knn.predict(x_test)

'''
4 模型評估
'''
print("準確率:", knn.score(x_test, y_test))
print("其他指標：\n", classification_report(
    y_test, y_predict, target_names=news.target_names))

from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import train_test_split
# 導入文本特徵矢量轉化模塊
from sklearn.feature_extraction.text import CountVectorizer
# 導入樸素貝葉斯模型
from sklearn.naive_bayes import MultinomialNB
# 模型評估模塊
from sklearn.metrics import classification_report

'''
樸素貝葉斯模型廣泛用於海量互聯網文本分類任務。
由於假設特徵條件相互獨立，預測需要估計的參數規模從冪指數量級下降接近線性量級，節約內存和計算時間
但是 該模型無法將特徵之間的聯繫考慮，數據關聯較強的分類任務表現不好。
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
3 貝葉斯分類器對新聞進行預測
'''
# 進行文本轉化為特徵
vec = CountVectorizer()
x_train = vec.fit_transform(x_train)
x_test = vec.transform(x_test)
# 初始化樸素貝葉斯模型
mnb = MultinomialNB()
# 訓練集合上進行訓練， 估計參數
mnb.fit(x_train, y_train)
# 對測試集合進行預測 保存預測結果
y_predict = mnb.predict(x_test)

'''
4 模型評估
'''
print("準確率:", mnb.score(x_test, y_test))
print("其他指標：\n", classification_report(
    y_test, y_predict, target_names=news.target_names))

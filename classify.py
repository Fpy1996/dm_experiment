import time
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

#以sklearn内置的20Newsgroups作为数据集, 类别为categories
news_data = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)

#拆分数据集为训练集和测试集，其中测试集占0.25
x_train, x_test, y_train, y_test = train_test_split(news_data.data, news_data.target, test_size=0.25, random_state=33)

#去除停用词，对所有训练集和测试集数据进行tf-idf特征抽取，结合CountVectorizer与TfidfTransformer
count_vect = CountVectorizer(analyzer='word', stop_words='english')
tfidf_transformer = TfidfTransformer()

x_train_counts = count_vect.fit_transform(x_train)
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

x_test_counts = count_vect.transform(x_test)
x_test_tfidf = tfidf_transformer.transform(x_test_counts)

def model_evaluation(model, x_train, x_test, y_train, y_test):
    """
    模型评估函数，输出对模型的评估结果：其在训练集、测试集上的准确率、AUC值，以及召回率、精度、F1值等
    :param model:所使用的模型
    :param x_train: 训练样本的特征向量列表
    :param x_test: 测试样本的特征向量列表
    :param y_train: 训练样本的类别列表
    :param y_test: 测试样本的类别列表
    """
    # 计算模型在训练集上的准确率与AUC值
    train_accuracy_rate = model.score(x_train, y_train)
    train_fpr, train_tpr, train_thre = metrics.roc_curve(y_train, model.predict_proba(x_train)[:, 1], pos_label=1)
    train_auc = metrics.auc(train_fpr, train_fpr)
    print("在训练集上的准确率={0}".format(train_accuracy_rate))
    print("在训练集上的AUC值={0}\n".format(train_auc))

    # 计算模型在测试集上的准确率与AUC值
    test_accuracy_rate = model.score(x_test, y_test)
    test_fpr, test_tpr, test_thre = metrics.roc_curve(y_test, model.predict_proba(x_test)[:, 1], pos_label=1)
    test_auc = metrics.auc(test_fpr, test_tpr)
    print("在测试集上的准确率={0}".format(test_accuracy_rate))
    print("在测试集上的AUC值={0}\n".format(test_auc))

def deal_report(classification_report):
    lines = classification_report.split('\n')
    lines = lines[2:len(lines)-5]
    precisions, recalls, f1_scores = [], [], []
    for line in lines:
        row_data = line.split('      ')
        precisions.append(float(row_data[2]))
        recalls.append(float(row_data[3]))
        f1_scores.append(float(row_data[4]))
    print("精度={0}，召回率={1}，f1分数={2}".format(np.mean(precisions), np.mean(recalls), np.mean(f1_scores)))


"""
使用朴素贝叶斯分类器进行文本分类
"""
print("\n============使用朴素贝叶斯分类器进行分类============")
#训练朴素贝叶斯分类模型，拉普拉斯平滑系数设置为1
start_time = time.process_time()
mnb_classify = MultinomialNB(alpha=1)
mnb_classify.fit(x_train_tfidf, y_train)
print("模型训练时间={0}s".format(time.process_time() - start_time))
#对测试集进行预测
start_time = time.process_time()
y_mnb_predict = mnb_classify.predict(x_test_tfidf)
print("模型对测试集分类时间={0}s".format(time.process_time() - start_time))
#进行模型评估
model_evaluation(mnb_classify, x_train_tfidf, x_test_tfidf, y_train, y_test)
#print(metrics.classification_report(y_test, y_mnb_predict))
mnb_report = metrics.classification_report(y_test, y_mnb_predict)
print(mnb_report)
deal_report(mnb_report)


"""
使用随机森林进行文本分类，测试子分类器个数为20和100的情况
"""
print("\n===========利用随机森林进行分类============")
#训练随机森林分类器，子分类器数量为20个
print("---子分类器为20个")
start_time = time.process_time()
rf_20_classify = RandomForestClassifier(n_estimators=20, criterion='gini', max_features="auto", bootstrap=True)
rf_20_classify.fit(x_train_tfidf, y_train)
print("模型训练时间={0}s".format(time.process_time() - start_time))
#对测试集进行预测
start_time = time.process_time()
y_rf_20_predict = rf_20_classify.predict(x_test_tfidf)
print("模型对测试集分类时间={0}s".format(time.process_time() - start_time))
#进行模型评估
model_evaluation(rf_20_classify, x_train_tfidf, x_test_tfidf, y_train, y_test)
rf_20_report = metrics.classification_report(y_test, y_rf_20_predict)
print(rf_20_report)
deal_report(rf_20_report)

#训练随机森林分类器，子分类器数量为200个
print("\n---子分类器为200个")
start_time = time.process_time()
rf_200_classify = RandomForestClassifier(n_estimators=200, criterion='gini', max_features="auto", bootstrap=True)
rf_200_classify.fit(x_train_tfidf, y_train)
print("模型训练时间={0}s".format(time.process_time() - start_time))
#对测试集进行预测
start_time = time.process_time()
y_rf_200_predict = rf_200_classify.predict(x_test_tfidf)
print("模型对测试集分类时间={0}s".format(time.process_time() - start_time))
#进行模型评估
model_evaluation(rf_200_classify, x_train_tfidf, x_test_tfidf, y_train, y_test)
rf_200_report = metrics.classification_report(y_test, y_rf_200_predict)
print(rf_200_report)
deal_report(rf_200_report)
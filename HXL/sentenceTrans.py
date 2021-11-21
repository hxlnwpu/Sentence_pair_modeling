from sentence_transformers import SentenceTransformer, util
import os
import csv
from sklearn import metrics


def cal_acc(result_dir ="./result_file"):
    accuracy_file = os.path.join(result_dir, 'accuracy.txt')
    # 存放结果的文件
    trainResult = os.path.join(result_dir, 'trainBertResult.csv')
    testResult = os.path.join(result_dir, 'testBertResult.csv')
    # 计算accuracy
    true_label_list = list()
    pre_label_list = list()
    with open(accuracy_file, 'a') as f:
        with open(trainResult, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.reader(csvfile)
            for index, row in enumerate(reader):
                if index == 0:
                    continue
                true_label_list.append(int(row[3]))
                pre_label_list.append(int(row[4]))
            f.write("------------------train - ----------------------"+'\n')
            f.write("train accuracy: " + str(metrics.accuracy_score(true_label_list, pre_label_list)) + '\n')
            f.write("precision_score: " + str(metrics.precision_score(true_label_list, pre_label_list)) + '\n')
            f.write("recall_score: " + str(metrics.recall_score(true_label_list, pre_label_list)) + '\n')
            f.write("f1_score: " + str(metrics.f1_score(true_label_list, pre_label_list)) + '\n')
        print("------------------train-----------------------")
        print("accuracy_score: " + str(metrics.accuracy_score(true_label_list, pre_label_list)))
        print("precision_score: " + str(metrics.precision_score(true_label_list, pre_label_list)))
        print("recall_score: " + str(metrics.recall_score(true_label_list, pre_label_list)))
        print("f1_score: " + str(metrics.f1_score(true_label_list, pre_label_list)))

        true_label_list.clear()
        pre_label_list.clear()
        with open(testResult, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.reader(csvfile)
            for index, row in enumerate(reader):
                if index == 0:
                    continue
                true_label_list.append(int(row[3]))
                pre_label_list.append(int(row[4]))
            f.write("------------------test ------------------------"+'\n')
            f.write("test accuracy: " + str(metrics.accuracy_score(true_label_list, pre_label_list)) + '\n')
            f.write("precision_score: " + str(metrics.precision_score(true_label_list, pre_label_list)) + '\n')
            f.write("recall_score: " + str(metrics.recall_score(true_label_list, pre_label_list)) + '\n')
            f.write("f1_score: " + str(metrics.f1_score(true_label_list, pre_label_list)) + '\n')
        print("------------------test-----------------------")
        print("accuracy_score: " + str(metrics.accuracy_score(true_label_list, pre_label_list)))
        print("precision_score: " + str(metrics.precision_score(true_label_list, pre_label_list)))
        print("recall_score: " + str(metrics.recall_score(true_label_list, pre_label_list)))
        print("f1_score: " + str(metrics.f1_score(true_label_list, pre_label_list)))

def get_bert_sim1(data_dir = "../BQ Corpus/data",result_dir ="./result_file",model_name = 'distiluse-base-multilingual-cased-v1' ):
    train_data = os.path.join(data_dir,"train.csv")
    test_data = os.path.join(data_dir,"test.csv")
    trainResult = os.path.join(result_dir, "trainBertResult.csv")
    testResult = os.path.join(result_dir, "testBertResult.csv")

    model = SentenceTransformer(model_name)
    # 获取train预测结果
    train_sample_num = 0
    sentences1 = list()
    sentences2 = list()
    true_labels = list()
    with open(train_data, 'r', encoding='utf-8-sig') as csvfile:  # 获取train所有句子
        reader = csv.reader(csvfile)
        for index, row in enumerate(reader):
            if index == 0:
                continue
            sentence1 = row[0]
            sentence2 = row[1]
            label = int(row[2])
            sentences1.append(sentence1)
            sentences2.append(sentence2)
            true_labels.append(label)
            train_sample_num += 1
    with open(test_data, 'r', encoding='utf-8-sig') as csvfile:  # 获取test所有句子
        reader = csv.reader(csvfile)
        for index, row in enumerate(reader):
            if index == 0:
                continue
            sentence1 = row[0]
            sentence2 = row[1]
            sentences1.append(sentence1)
            sentences2.append(sentence2)
            true_labels.append(label)

    # Compute embedding for both lists
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)

    # Compute cosine-similarits
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

    with open(trainResult, "w+", encoding='utf-8-sig', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "sentence1", "sentence2", "true_label", "label", "similarity"])
        for i in range(0,train_sample_num):
            id = i
            sentence1 = sentences1[i]
            sentence2  =sentences2[i]
            true_label = true_labels[i]
            similarity = cosine_scores[i][i]
            label = 1 if similarity >= 0.5 else 0
            writer.writerow([id, sentence1, sentence2, true_label, label, similarity])

    with open(testResult, "w+", encoding='utf-8-sig', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "sentence1", "sentence2", "true_label", "label", "similarity"])
        for i in range(train_sample_num, len(true_labels)):
            id = i
            sentence1 = sentences1[i]
            sentence2 = sentences2[i]
            true_label = true_labels[i]
            similarity = cosine_scores[i]
            label = 1 if similarity >= 0.5 else 0
            writer.writerow([id, sentence1, sentence2, true_label, label, similarity])


if __name__ == '__main__':
    get_bert_sim1()
    cal_acc()
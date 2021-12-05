import pickle

from run_Bert_model import model_train_validate_test,model_load_test
from utils import Metric, json2df,csv2df
import os


# BQ Corpus
# if __name__== '__main__':
#     bq_path = "E:/桌面/sentence_pair_modeling/BQ Corpus"
#     train_df = csv2df(os.path.join(bq_path, "data/train.csv"))
#     dev_df = csv2df(os.path.join(bq_path, "data/dev.csv"))
#     test_df = csv2df(os.path.join(bq_path, "data/test.csv"))
#     target_dir = os.path.join(bq_path, "output/Bert/")
#     test_prediction_dir = 'E:/桌面/sentence_pair_modeling/BQ Corpus/output/Bert/'
#     test_prediction_name = 'train_prediction.csv'
#     # model_train_validate_test(train_df, dev_df, test_df, target_dir)
#     model_load_test(train_df, target_dir, test_prediction_dir, test_prediction_name)
#     print("end")

# LCQMC
if __name__== '__main__':
    lcqmc_path = "E:/桌面/sentence_pair_modeling/LCQMC"
    train_df = pd.read_csv(os.path.join(lcqmc_path, "data/train.tsv"),sep='\t',header=None, names=['s1','s2','label'])
    dev_df = pd.read_csv(os.path.join(lcqmc_path, "data/dev.tsv"),sep='\t',header=None, names=['s1','s2','label'])
    test_df = pd.read_csv(os.path.join(lcqmc_path, "data/test.tsv"),sep='\t',header=None, names=['s1','s2','label'])
    data = pd.concat([test_df]).reset_index(drop=True)

    target_dir = os.path.join(lcqmc_path, "output/Bert") # load pretrained model
    test_prediction_dir = os.path.join(lcqmc_path, "output/Bert") # where to save the infer result
    test_prediction_name = 'test_prediction_lcqmc.csv' # the infer result name

    model_load_test(test_df = data,
                    target_dir = target_dir,
                    test_prediction_dir = test_prediction_dir,
                    test_prediction_name = test_prediction_name)

    test_result = pd.read_csv(os.path.join(test_prediction_dir, test_prediction_name))




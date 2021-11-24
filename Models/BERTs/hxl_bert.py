import pickle

from run_Bert_model import model_train_validate_test,model_load_test
from utils import Metric, json2df,csv2df
import os





if __name__== '__main__':
    bq_path = "../../BQ Corpus/"
    train_df = csv2df(os.path.join(bq_path, "data/train.csv"))
    dev_df = csv2df(os.path.join(bq_path, "data/dev.csv"))
    test_df = csv2df(os.path.join(bq_path, "data/test.csv"))
    target_dir = os.path.join(bq_path, "output/Bert/")
    # model_train_validate_test(train_df, dev_df, test_df, target_dir,
    #         max_seq_len=64,
    #         num_labels=2,
    #         epochs=3,
    #         batch_size=32,
    #         lr=2e-05,
    #         patience=1,
    #         max_grad_norm=10.0,
    #         if_save_model=True,
    #         checkpoint=None)
    #
    # test_result = pd.read_csv(os.path.join(target_dir, 'test_prediction.csv'))
    # Metric(test_df.label, test_result.prediction)
    test_prediction_dir = os.path.join(bq_path, "output/Bert/")
    test_prediction_name = 'test_prediction.csv'

    # bert_embedding = BertModel.from_pretrained('./bert-base-chinese')

    # model_train_validate_test(train_df, dev_df, test_df, target_dir)
    model_load_test(test_df, target_dir, test_prediction_dir, test_prediction_name)

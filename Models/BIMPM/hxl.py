#run BiPIM
from run_BIMPM_model import model_train_validate_test
import pandas as pd
from utils import Metric, json2df,csv2df
import os

bq_path = "../../BQ Corpus/"
train_df = csv2df(os.path.join(bq_path, "data/train.csv"))
dev_df =  csv2df(os.path.join(bq_path, "data/dev.csv"))
test_df =  csv2df(os.path.join(bq_path, "data/test.csv"))

vocab_file = os.path.join(bq_path, "data/rand_word_vocab.txt")

target_dir = os.path.join(bq_path, "output/BIMPM_word_rand/")


if __name__ == '__main__':
    # 微调并且测试
    model_train_validate_test(train_df = train_df,
                dev_df = dev_df,
                test_df = test_df,
                embeddings_file = None,
                vocab_file = vocab_file,
                target_dir = target_dir,
                mode = 'word',
                num_labels=2,
                max_length=64,
                epochs=50,
                batch_size=128,
                lr=0.005,
                patience=3,
                max_grad_norm=10.0,
                gpu_index=0,
                if_save_model=True,
                checkpoint=None)
    
    test_result = pd.read_csv(os.path.join(target_dir, 'test_prediction.csv'))
    Metric(test_df.label, test_result.prediction) 
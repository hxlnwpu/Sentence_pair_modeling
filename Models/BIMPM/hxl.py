#run BiPIM
from run_BIMPM_model import model_train_validate_test,model_load_test
import pandas as pd
from utils import Metric, json2df,csv2df
import os

bq_path = "../../BQ Corpus/"
train_df = csv2df(os.path.join(bq_path, "data/train.csv"))
dev_df =  csv2df(os.path.join(bq_path, "data/dev.csv"))
test_df =  csv2df(os.path.join(bq_path, "data/test.csv"))

vocab_file = os.path.join(bq_path, "data/rand_word_vocab.txt")

target_dir = os.path.join(bq_path, "output/BIMPM_word_rand/")

model_file = os.path.join(bq_path,"output/BIMPM_word_rand/best.pth.tar")


def train_and_test():
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

def test():
    #使用微调好的模型测试
    model_load_test(test_df = test_df, 
                    vocab_file = vocab_file, 
                    embeddings_file = None, 
                    pretrained_file =model_file,
                    test_prediction_dir = target_dir, 
                    test_prediction_name = 'test_prediction.csv', 
                    mode = 'word', 
                    num_labels=2, 
                    max_length=64, 
                    gpu_index=0, 
                    batch_size=128)



if __name__ == '__main__':
    test()


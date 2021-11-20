from run_Xlnet_model import model_train_validate_test
import pandas as pd
from utils import Metric, json2df,csv2df
import os

bq_path = "../../BQ Corpus/"
train_df = csv2df(os.path.join(bq_path, "data/train.csv"))
dev_df =  csv2df(os.path.join(bq_path, "data/dev.csv"))
test_df =  csv2df(os.path.join(bq_path, "data/test.csv"))
target_dir = os.path.join(bq_path, "output/Xlnet/")

if __name__ == '__main__':
    model_train_validate_test(train_df, dev_df, test_df, target_dir, 
            max_seq_len=64, 
            num_labels=2,  
            epochs=10,
            batch_size=32,
            lr=2e-05,
            patience=1,
            max_grad_norm=10.0,
            if_save_model=True,
            checkpoint=None)

    test_result = pd.read_csv(os.path.join(target_dir, 'test_prediction.csv'))
    Metric(test_df.label, test_result.prediction)
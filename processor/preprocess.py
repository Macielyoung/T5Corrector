import pandas as pd
from datasets import Dataset


correction_data = "../dataset/correction_pair.csv"
correction_df = pd.read_csv(correction_data)
print(correction_df.shape)

correction_df = correction_df[correction_df['source'].notnull()]
correction_df = correction_df[correction_df['target'].notnull()]
print(correction_df.shape)

# max_source_len = max([len(str(source)) for source in correction_df['source'].unique()])
# max_target_len = max([len(str(target)) for target in correction_df['target'].unique()])

# print(max_source_len)
# print(max_target_len)

correction_dataset = Dataset.from_pandas(correction_df)
correction_dataset = correction_dataset.remove_columns("Unnamed: 0")
print(correction_dataset)
correction_train_test_dataset = correction_dataset.train_test_split(test_size=0.1)
print(correction_train_test_dataset)

save_path = "../dataset/correction"
correction_train_test_dataset.save_to_disk(save_path)

# # save dataset to huggingface hub
# hub_path = ""
# correction_train_test_dataset.push_to_hub(hub_path)
import re
import pandas as pd


def contains_only_chinese_and_punctuation(text):
    # 匹配中文和标点符号的正则表达式
    pattern = re.compile(r'^[\u4e00-\u9fa5|，|。|！|？|、|；|“|”|‘|’|（|）]+$')
    return pattern.match(text) is not None
    
    
correction_file = "../dataset/correction_pair.csv"
correction_df = pd.read_csv(correction_file)
correction_df = correction_df.fillna("")
target = list(correction_df['target'].unique())
target_lines = [line for line in target if contains_only_chinese_and_punctuation(line)]

target_items = [{'passage': text} for text in target_lines]
target_file = "../dataset/correction_texts.csv"
target_df = pd.DataFrame(target_items)
target_df.to_csv(target_file)
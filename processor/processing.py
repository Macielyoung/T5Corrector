import pandas as pd
import numpy as np
import math
import json
from collections import defaultdict
from datasets import load_dataset, Dataset, concatenate_datasets
import random
from zhconv import convert
from LAC import LAC
# from pronunciating import PronunciationRetrieval
from random import shuffle
import copy


MAX_SAMPLE_NUM = 3
MAX_NGRAM_SAMPLE_NUM = 3
MAX_SHUFFLE_NUM = 4
SAMPLE_RATIO = 0.1
SHUFFLE_SAMPLE_RATIO = 0.15
SAME_TOPK = 20
SIMILAR_TOPK = 10
seg = LAC(mode='seg')


def read_stock_list(stock_file):
    with open(stock_file, 'r') as f:
        stocks = [line.strip() for line in f.readlines()]
    return stocks


def read_confused_chinese(confused_file):
    confused_chinese_relation = {}
    with open(confused_file, 'r') as f:
        data = json.load(f)
        for key, value in data.items():
            same_pronunciation_chars = value['same_pronunciation_chars']
            same_pronunciation_chars = [item['char'] for item in same_pronunciation_chars]
            similar_pronunciation_chars = value['similar_pronunciation_chars']
            similar_pronunciation_chars = [item['char'] for item in similar_pronunciation_chars]
            similar_font_chars = value['similar_font_chars']
            similar_font_chars = [item['char'] for item in similar_font_chars]
            relation_items = {'same_pronunciation_chars': same_pronunciation_chars,
                              'similar_pronunciation_chars': similar_pronunciation_chars,
                              'similar_font_chars': similar_font_chars}
            confused_chinese_relation[key] = relation_items
    return confused_chinese_relation
            

def replace_confused_chinese(text, confused_chinese_relation):
    # 5% 句子不替换
    text_replaced_prob = random.random()
    # print(text_replaced_prob)
    if text_replaced_prob < 0.05:
        replaced_text = text
    else:
        word_replaced_prob = np.random.rand(len(text)).tolist()
        # print(word_replaced_prob)
        replaced_text = ""
        for char, prob in zip(text, word_replaced_prob):
            if char in confused_chinese_relation:
                # 5%的字使用同音字替换
                if prob <= 0.05:
                    same_pronunciation_chars = confused_chinese_relation[char]['same_pronunciation_chars']
                    if len(same_pronunciation_chars) > 0:
                        replaced_char = random.choice(same_pronunciation_chars)
                        replaced_text += replaced_char
                    else:
                        replaced_text += char
                # 5%的字使用近音字替换
                elif prob <= 0.1:
                    similar_pronunciation_chars = confused_chinese_relation[char]['similar_pronunciation_chars']
                    if len(similar_pronunciation_chars) > 0:
                        replaced_char = random.choice(similar_pronunciation_chars)
                        replaced_text += replaced_char
                    else:
                        replaced_text += char
                # 5%的字使用近形字替换
                elif prob <= 0.15:
                    similar_font_chars = confused_chinese_relation[char]['similar_font_chars']
                    if len(similar_font_chars) > 0:
                        replaced_char = random.choice(similar_font_chars)
                        replaced_text += replaced_char
                    else:
                        replaced_text += char
                # 85%的字不替换
                else:
                    replaced_text += char
            else:
                replaced_text += char
    return replaced_text


def replace_confused_char(text, confused_chinese_relation):
    # 以字为单位进行干扰词替换
    
    # 2% 句子不替换
    text_replaced_prob = random.random()
    # print(text_replaced_prob)
    if text_replaced_prob < 0.02:
        replaced_text = text
    else:
        # 抽样比例 $SAMPLE_RATIO$, 最大抽样数量 $MAX_SAMPLE_NUM$
        chars_num = len(text)
        sample_num = math.ceil(chars_num * SAMPLE_RATIO)
        selected_num = min(sample_num, MAX_SAMPLE_NUM)
        sample_indices = random.sample(range(chars_num), selected_num)
        # print(sample_indices)
        
        words_list = list(text)
        for idx in sample_indices:
            char = words_list[idx]
            if char in confused_chinese_relation:
                prob = random.random()
                # 1/3的字使用同音字替换
                if prob <= 0.33:
                    same_pronunciation_chars = confused_chinese_relation[char]['same_pronunciation_chars']
                    if len(same_pronunciation_chars) > 0:
                        replaced_char = random.choice(same_pronunciation_chars)
                        words_list[idx] = replaced_char
                # 1/3的字使用近音字替换
                elif prob <= 0.66:
                    similar_pronunciation_chars = confused_chinese_relation[char]['similar_pronunciation_chars']
                    if len(similar_pronunciation_chars) > 0:
                        replaced_char = random.choice(similar_pronunciation_chars)
                        words_list[idx] = replaced_char
                # 1/3的字使用近形字替换
                else:
                    similar_font_chars = confused_chinese_relation[char]['similar_font_chars']
                    if len(similar_font_chars) > 0:
                        replaced_char = random.choice(similar_font_chars)
                        words_list[idx] = replaced_char
        replaced_text = "".join(words_list)
    return replaced_text


def replace_confused_word(text, ngram):
    # 以词为单位进行干扰词替换
    words = seg.run(text)
    ngram_word_locs = [idx for idx, word in enumerate(words) if len(word) == ngram]
    if len(ngram_word_locs) > 0:
        # 抽样比例 $SAMPLE_RATIO$, 最大抽样数量 $MAX_SAMPLE_NUM$
        ngram_words_num = len(ngram_word_locs)
        sample_num = math.ceil(ngram_words_num * SAMPLE_RATIO)
        selected_num = min(sample_num, MAX_NGRAM_SAMPLE_NUM)
        sample_indices = random.sample(range(ngram_words_num), selected_num)
        
        ngram_loc = [ngram_word_locs[index] for index in sample_indices]
        for loc in ngram_loc:
            word = words[loc]
            prob = random.random()
            if prob <= 0.8:
                # 4/5的字使用同音字替换
                pronunciations = pronunciation_retrieval.convert_word_to_pronunciation(word)
                same_pronunciation_words = pronunciation_retrieval.get_same_pronunciation_word(pronunciations, SAME_TOPK)
                same_pronunciation_words = pronunciation_retrieval.filter_same_word(same_pronunciation_words, word)
                if len(same_pronunciation_words) > 0:
                    replace_word = random.choice(same_pronunciation_words)
                else:
                    replace_word = word
            else:
                # 1/5的字使用近音字替换
                pronunciations = pronunciation_retrieval.convert_word_to_pronunciation(word)
                similar_pronunciations = pronunciation_retrieval.get_similar_pronunciations(pronunciations)
                similar_pronunciation_words = pronunciation_retrieval.get_similar_pronunciation_words(similar_pronunciations, SIMILAR_TOPK)
                if len(similar_pronunciation_words) > 0:
                    replace_word = random.choice(similar_pronunciation_words)
                else:
                    replace_word = word
            words[loc] = replace_word
        replace_text = "".join(words)
        return replace_text
    else:
        return None


def del_confused_word(words):
    # 在 原 始 句 子 中 删 除 部 分 词 组 中 的 词
    word_num = len([word for word in words if len(word) > 1])
    delelte_num = math.ceil(word_num * SAMPLE_RATIO)
    selected_num = min(delelte_num, MAX_DELETE_NUM)

    ngram_word_indices = [wid for wid, word in enumerate(words) if len(word) > 1]
    selected_indices = random.sample(ngram_word_indices, selected_num)

    replace_words = copy.deepcopy(words)
    for index in selected_indices:
        selected_word = replace_words[index]
        chars_list = list(selected_word)
        del_index = random.choice(range(len(chars_list)))
        chars_list.pop(del_index)
        replace_word = "".join(chars_list)
        replace_words[index] = replace_word
    replace_text = "".join(replace_words)
    return replace_text


def shuffle_confused_words(words):
    # 随机打乱ngram的词序
    word_num = len([word for word in words if len(word) > 1])
    shuffle_num = math.ceil(word_num * SHUFFLE_SAMPLE_RATIO)
    selected_num = min(shuffle_num, MAX_SHUFFLE_NUM)

    ngram_word_indices = [wid for wid, word in enumerate(words) if len(word) > 1]
    selected_indices = random.sample(ngram_word_indices, selected_num)

    replace_words = copy.deepcopy(words)
    for index in selected_indices:
        selected_word = replace_words[index]
        chars_list = list(selected_word)

        if len(chars_list) == 2:
            chars_list = [chars_list[1], chars_list[0]]
        else:
            shuffle(chars_list)
        replace_word = "".join(chars_list)
        replace_words[index] = replace_word
    replace_text = "".join(replace_words)
    return replace_text
    

# 加载同音字和近音字检索模块
common_file = "../rawdata/chinese_3500.txt"
chinese_pronunciation_file = "../rawdata/chinese_pronunciation.txt"
pronunciation_retrieval = PronunciationRetrieval(common_file, chinese_pronunciation_file)

# 读取同音字、近音字、近形字配置文件
confused_file = "../rawdata/confused_chinese.json"
confused_chinese_relation = read_confused_chinese(confused_file)

# 股票名称作为专业名称进行分词，加入固定此表中
stock_file = "../rawdata/stock.txt"
seg.load_customization(stock_file, sep=None)

correction_path = "../dataset/correction_texts.csv"
correction_df = pd.read_csv(correction_path)
correction_dataset = Dataset.from_pandas(correction_df)
correction_dataset = correction_dataset.remove_columns("Unnamed: 0")
print("read correction data successfully")

stock_question_path = "../dataset/stock_passage.csv"
stock_question_df = pd.read_csv(stock_question_path)
stock_question_dataset = Dataset.from_pandas(stock_question_df)
stock_question_dataset = stock_question_dataset.remove_columns("Unnamed: 0")
print("read stock question data successfully")

passage_dataset = load_dataset('beyond/chinese_clean_passages_80m')
passage_dataset = passage_dataset['train'].filter(lambda example: len(example['passage']) <= 100)
N = len(passage_dataset)
k = 5000000
passage_dataset = passage_dataset.select(random.sample(range(N), k=k))
print("read passage data successfully")

# 合并三个数据集
all_dataset = concatenate_datasets([passage_dataset, correction_dataset, stock_question_dataset], axis=0)


def add_confused_to_text(examples):
    """
    对于原始样本增加扰动，构建错误样本到正确样本的映射对
    """
    res = defaultdict(list)
    passages = examples['passage']
    for p in passages:
        # 针对部分繁体字，先做文字简写
        p = convert(p, 'zh-cn')
        # passage:
        res['target'].append(p)
        source = replace_confused_char(p, confused_chinese_relation)
        res['source'].append(source)
        
        # 替换ngram的汉词
        ngrams = [2, 2, 2, 3, 3, 4]
        for ngram in ngrams:
            ngram_source = replace_confused_word(p, ngram)
            if ngram_source is not None:
                res['target'].append(p)
                res['source'].append(ngram_source)

        # 打乱ngram的汉词
        words = seg.run(p)
        shuffle_times = 2
        for _ in range(shuffle_times):
            shuffle_source = shuffle_confused_words(words)
            res['target'].append(p)
            res['source'].append(shuffle_source)
    return res


dataset_with_confusing = all_dataset.map(add_confused_to_text, 
                                         batched=True, 
                                         batch_size=50, 
                                         num_proc=20,
                                         remove_columns=passage_dataset.column_names)
dataset_with_confusing = dataset_with_confusing.train_test_split(test_size=0.01)

print(dataset_with_confusing)
dataset_with_confusing.save_to_disk(f'../confused_chinese/correction')

# -*- coding: utf-8 -*-
# @Time    : 2023/1/30
# @Author  : Maciel


from Pinyin2Hanzi import dag, DefaultDagParams
from pypinyin import pinyin, Style, load_single_dict
import copy


class PronunciationRetrieval:
    def __init__(self, common_file, chinese_pronunciation_file):
        self.dagparams = DefaultDagParams()
        self.common_chars = self.read_common_chars(common_file)
        self.chinese_pronunciation = self.read_chinese_pronunciation(chinese_pronunciation_file)
        # self.load_common_pronunciation()
        self.common_pronunciations = self.fetch_common_pronunciations()
        
        
    
    def read_chinese_pronunciation(self, chinese_pronunciation_file):
        chinese_pronunciation = {}
        with open(chinese_pronunciation_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                arr = line.split(":")[-1].strip()
                pronunciations_char_pair = arr.split("#")
                if len(pronunciations_char_pair) == 2:
                    pronunciations, char = pronunciations_char_pair
                    pronunciations = pronunciations.strip()
                    char = char.strip()
                    if char == '""':
                        continue
                    chinese_pronunciation[char] = pronunciations
        return chinese_pronunciation
                
        
    def read_common_chars(self, common_file):
        # 读取常用字
        with open(common_file, "r", encoding="utf-8") as f:
            common_chars = list(f.read().strip())
            common_chars = common_chars[1:]
        return common_chars
    
    
    def load_common_pronunciation(self):
        # 加载常用字的发音
        for char in self.common_chars:
            char_pronunciations = self.chinese_pronunciation.get(char, "")
            if char_pronunciations == "":
                continue
            load_single_dict({ord(char): char_pronunciations})
            
    
    def fetch_common_pronunciations(self):
        # 获取常用字的全部发音
        common_pronunciations = []
        for char in self.common_chars:
            char_pronunciations = self.convert_word_to_pronunciation(char)
            common_pronunciations.extend(char_pronunciations)
        common_pronunciations = list(set(common_pronunciations))
        return common_pronunciations
        
        
    def convert_pronunciation_to_word(self, pronunciation, path_num):
        # 获取该发音的topk个汉词
        results = dag(self.dagparams, pronunciation, path_num=path_num)
        items = []
        for item in results:
            word_list = item.path
            word = "".join(word_list)
            items.append(word)
        return items
    
    
    def convert_word_to_pronunciation(self, word):
        # 将汉字转化为拼音
        if word == "嗯":
            pronunciations = ['en']
        else:
            pronunciations = pinyin(word, style=Style.TONE3, heteronym=True)
            pronunciations = [proun[0][:-1] for proun in pronunciations]
        return pronunciations
    
    
    def get_same_pronunciation_word(self, pronunciations, topk):
        # 获取同音字
        same_pronunciation_words = []
        char_items = self.convert_pronunciation_to_word(pronunciations, topk)
        same_pronunciation_words.extend(char_items)
        return same_pronunciation_words
    
    
    def filter_same_word(self, same_pronunciation_words, word):
        # 过滤本体词
        same_pronunciation_words = [item for item in same_pronunciation_words if item != word]
        return same_pronunciation_words
    
    
    def get_similar_pronunciations(self, pronunciations):
        # 获取相近拼音
        # 1. 前鼻音和后鼻音区分 (n vs ng)
        # 2. n 和 l 区分 (n vs l)
        # 3. f 和 h 区分 (f vs h)
        # 4. r 和 l 区分 (r vs l)
        # 5. s 和 sh 区分 (s vs sh)
        # 6. c 和 ch 区分 (c vs ch)
        # 7. z 和 zh 区分 (z vs zh)
        similar_pronunciation_dict = {'n': ['l', 'ng'],
                                      'ng': ['n'],
                                      'l': ['n', 'r'],
                                      'r': ['l'],
                                      's': ['sh'],
                                      'sh': ['s'],
                                      'c': ['ch'],
                                      'ch': ['c'],
                                      'z': ['zh'],
                                      'zh': ['z']}
        similar_pronunciations = []
        for pid, pronunciation in enumerate(pronunciations):
            for key in similar_pronunciation_dict.keys():
                if key in pronunciation:
                    for value in similar_pronunciation_dict[key]:
                        sim_pron = pronunciation.replace(key, value)
                        if sim_pron in self.common_pronunciations:
                            sim_pronunciation = copy.deepcopy(pronunciations)
                            sim_pronunciation[pid] = sim_pron
                            similar_pronunciations.append(sim_pronunciation)
        return similar_pronunciations
    
    
    def get_similar_pronunciation_words(self, similar_pronunciations, topk):
        # 获取近音字
        similar_pronunciation_words = []
        for pronunciations in similar_pronunciations:
            sim_pron_words = self.get_same_pronunciation_word(pronunciations, topk)
            similar_pronunciation_words += sim_pron_words
        return similar_pronunciation_words
         
    
if __name__ == "__main__":
    common_file = "../rawdata/chinese_3500.txt"
    chinese_pronunciation_file = "../rawdata/chinese_pronunciation.txt"
    pronunciation_retrieval = PronunciationRetrieval(common_file, chinese_pronunciation_file)
    
    word = "兴业银行"
    word = "投资"
    same_topk = 5
    pronunciations = pronunciation_retrieval.convert_word_to_pronunciation(word)
    print(pronunciations)
    
    # same_pronunciation_words = pronunciation_retrieval.get_same_pronunciation_word(pronunciations, same_topk)
    # same_pronunciation_words = pronunciation_retrieval.filter_same_word(same_pronunciation_words, word)
    # print(same_pronunciation_words)
    
    similar_pronunciations = pronunciation_retrieval.get_similar_pronunciations(pronunciations)
    print(similar_pronunciations)
    
    similar_topk = 3
    similar_pronunciation_words = pronunciation_retrieval.get_similar_pronunciation_words(similar_pronunciations, similar_topk)
    print(similar_pronunciation_words)
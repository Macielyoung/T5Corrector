# T5Corrector
#### 模型功能与结构

1. 该模型功能主要是中文文本纠错，模型结构基于[mengzi-t5-base](https://huggingface.co/Langboat/mengzi-t5-base)进行继续预训练。
2. 使用中文纯语料，通过替换同音字、近音字和形近词来构造错误—修正的平行语料库。具体方法可以参考[中文混淆字挖掘](https://github.com/Macielyoung/Confused_Chinese)的方法。
3. 预训练时句子经过分词，对其中ngram的词组进行全部替换来更好支持的词组的纠正。



#### 如何使用

```python
# 加载模型
from transformers import T5Tokenizer, T5ForConditionalGeneration
pretrained = "Maciel/T5Corrector-base-v1"
tokenizer = T5Tokenizer.from_pretrained(pretrained)
model = T5ForConditionalGeneration.from_pretrained(pretrained)

# 文本纠错推理
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def correct(text, max_length):
		model_inputs = tokenizer(text, 
                           	 max_length=max_length, 
                           	 truncation=True, 
                           	 return_tensors="pt").to(device)
    output = model.generate(**model_inputs, 
                              num_beams=5,
                              no_repeat_ngram_size=4,
                              do_sample=True, 
                              early_stopping=True,
                              max_length=max_length,
                              return_dict_in_generate=True,
                              output_scores=True)
    pred_output = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)[0]
    return pred_output

text = "听到这个消息，心情真的蓝瘦"
correction = correct(text, max_length=32)
print(correction)
```

我们在**huggingface**上提供了下载链接和体验接口：

| 模型                                                         | 支持语言 | 备注                                           |
| :----------------------------------------------------------- | :------- | :--------------------------------------------- |
| [T5Corrector](https://huggingface.co/Maciel/T5Corrector-base-v1) | 中文     | 选择500w中文文本，替换关联词，生成3kw+对照语料 |


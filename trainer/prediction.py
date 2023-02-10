import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BertTokenizer
import os
# import nltk
import readline
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


saved_model_path = "../models/checkpoint-9000"
tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(saved_model_path)
    
max_input_length = 64
while True:
    print("input your sentence: ...")
    sentence = input()
    input_encodings = tokenizer(sentence, 
                                max_length=max_input_length, 
                                truncation=True, 
                                return_tensors="pt")
    if "token_type_ids" in input_encodings.keys():
        input_encodings.pop("token_type_ids")
    output = model.generate(**input_encodings, 
                            num_beams=5,
                            no_repeat_ngram_size=4,
                            do_sample=True, 
                            early_stopping=True,
                            min_length=5, 
                            max_length=64,
                            return_dict_in_generate=True,
                            output_scores=True)
    decoded_output = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)[0]
    decoded_scores = output.sequences_scores
    confidence = torch.exp(decoded_scores).item()
    generation = decoded_output.strip()
    # generation = nltk.sent_tokenize(decoded_output.strip())[0]
    correction = generation.split("</s>")[0]
    print("sentence: {}\ncorrection: {}\nconfidence: {}\n".format(sentence, correction, confidence))

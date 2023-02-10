from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq, TrainingArguments, Trainer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_from_disk
import evaluate
import os
import re
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def process_data(examples):
    # T5 model
    source_max_length = 512
    target_max_length = 512
    model_inputs = tokenizer(examples['source'],
                             max_length=source_max_length,
                             padding=True,
                             truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['target'], 
                           max_length=target_max_length,
                           padding=True, 
                           truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


def remove_punctuation(text):
    punc = '~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'
    pure_text = re.sub(r"[%s]+" %punc, "", text)
    return pure_text
    

def compute_metrics(eval_pred):
    # calculate chinese rouge metric
    
    predictions, labels = eval_pred
    # print("predictions: {}, labels: {}".format(predictions, labels))
    # predictions = np.argmax(predictions, axis=1)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # calculate prediction accuracy
    decoded_transform_preds = [remove_punctuation(pred) for pred in decoded_preds]
    decoded_transform_labels = [remove_punctuation(label) for label in decoded_labels]
    equal_pred_label = [1 for pred, label in zip(decoded_transform_preds, decoded_transform_labels) if pred == label]
    accuracy = len(equal_pred_label) / len(decoded_labels)
    
    # Rouge expects a newline after each sentence
    decoded_preds = [" ".join(pred.strip()) for pred in decoded_preds]
    decoded_labels = [" ".join(label.strip()) for label in decoded_labels]
    
    # Compute ROUGE scores
    result = metric.compute(predictions=decoded_preds, 
                            references=decoded_labels,
                            tokenizer=lambda x: x.split())

    # Extract ROUGE f1 scores
    result = {key: round(value * 100, 4) for key, value in result.items()}
    
    # Add mean generated length to metrics
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id)
                       for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    result['gen_acc'] = round(accuracy, 4)
    return result


def computer_chinese_metrics(eval_preds):
    from chinese_rouge import Rouge
    rouge = Rouge()
    
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [list(pred) for pred in decoded_preds]    
    decoded_labels = [list(label) for label in decoded_labels]

    decoded_preds = [" ".join(pred_token) for pred_token in decoded_preds]
    decoded_labels = [" ".join(label_token) for label_token in decoded_labels]

    scores = rouge.get_scores(decoded_preds, decoded_labels)
    
    items = ['rouge-1', 'rouge-2', 'rouge-l']
    score = {}
    for item in items:
        item_score = []
        for score in scores:
            # print(score)
            item_score.append(score[item]['f'])
        mean_item_score = np.mean(item_score)
        score[item] = round(mean_item_score, 4)
    return score


pretrained_model = "Langboat/mengzi-t5-base"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)
data_collator = DataCollatorForSeq2Seq(tokenizer)
pretrained_model_name = pretrained_model.split("/")[-1]
print("pretrained_model_name: {}, load model and tokenizer done!".format(pretrained_model_name))

metric = evaluate.load("rouge")
print("load rouge metric done!")

correction_data_path = "../confused_chinese/correction"
dataset_with_correction = load_from_disk(correction_data_path)
train_correction_dataset = dataset_with_correction['train']
print(train_correction_dataset)
train_correction_dataset = train_correction_dataset.map(process_data,
                                                        batched=True,
                                                        remove_columns=train_correction_dataset.column_names)
eval_correction_dataset = dataset_with_correction['test']
print(eval_correction_dataset)
eval_correction_dataset = eval_correction_dataset.map(process_data,
                                                      batched=True,
                                                      remove_columns=eval_correction_dataset.column_names)


saved_dir = "../models/{}".format(pretrained_model_name)
train_batch_size = 16
eval_batch_size = 32
args = Seq2SeqTrainingArguments(
    output_dir=saved_dir, #The output directory
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=10, # number of training epochs
    per_device_train_batch_size=train_batch_size, # batch size for training
    per_device_eval_batch_size=eval_batch_size,  # batch size for evaluation
    eval_steps=3000, # Number of update steps between two evaluations.
    save_steps=3000, # after # steps model is saved 
    warmup_steps=1000,# number of warmup steps for learning rate scheduler
    # prediction_loss_only=True,
    predict_with_generate=True,
    save_total_limit=3,
    load_best_model_at_end=True,
    save_strategy="steps",
    evaluation_strategy="steps",
    gradient_accumulation_steps=4,
    learning_rate=5e-4,
    weight_decay=0.01,
    fp16=True,
    report_to="tensorboard"
)

trainer = Seq2SeqTrainer(
    model,
    args=args,
    train_dataset=train_correction_dataset,
    eval_dataset=eval_correction_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("start train model...")
trainer.train()
trainer.save_model()
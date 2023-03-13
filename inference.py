from datasets import load_metric
metric = load_metric("rouge")

input_lines = []
label_lines = []
with open(f'{task}/test.tsv') as file:
  for line in file:
    line = line.strip().split('\t')
    input = line[0]
    input_lines.append(input +'')
    label_lines.append(line[1])



input_lines  = input_lines
label_lines = label_lines
dict_obj = {'inputs': input_lines, 'labels': label_lines}

dataset = Dataset.from_dict(dict_obj)
test_tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=['inputs'], num_proc=10)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")

model = AutoModelForSeq2SeqLM.from_pretrained("/content/tmp/checkpoint-85675")
model.to('cuda')

import torch 
import numpy as np
metrics = load_metric('rouge')

max_target_length = 256
dataloader = torch.utils.data.DataLoader(test_tokenized_datasets, collate_fn=data_collator, batch_size=32)

predictions = []
references = []
for i, batch in enumerate(tqdm(dataloader)):
  outputs = model.generate(
      input_ids=batch['input_ids'].to('cuda'),
      max_length=max_target_length,
      attention_mask=batch['attention_mask'].to('cuda'),
  )
  with tokenizer.as_target_tokenizer():
    outputs = [tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out in outputs]

    labels = np.where(batch['labels'] != -100,  batch['labels'], tokenizer.pad_token_id)
    actuals = [tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out in labels]
  predictions.extend(outputs)
  references.extend(actuals)
  metrics.add_batch(predictions=outputs, references=actuals)


metrics.compute()

print([{k: v.mid.fmeasure} for k,v in metrics.compute(predictions=predictions, references=references).items()])

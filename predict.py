import time
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

# Load the checkpoint
checkpoint = torch.load('model/model_epoch_4.pth')

model = BERTClassifier(bert_model_name, num_classes, unique_list)

# Load the state_dict into the model
model.load_state_dict(checkpoint)

# Move the model to the appropriate device
if torch.cuda.is_available():
    model = model.cuda()
tokenizer = BertTokenizer.from_pretrained(bert_model_name)

x = time.time()
source = SourceFormatterAndDFG("test.sol")
source.read_input_file()
source.clean_source_code()
source.format_source_code()
test_text = source.source_code
print(test_text)
sentiment = predict_sentiment(test_text, model, tokenizer, device)
print(f"Predicted : {sentiment}")
print(f'Time: {time.time() - x}')
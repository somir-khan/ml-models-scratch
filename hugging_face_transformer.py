import pandas as pd
# Load the data from the path
data_path = "datasets/news_articles.csv"
news_data = pd.read_csv(data_path, on_bad_lines="skip")

# Show data information
# news_data.info()
from transformers import MarianTokenizer, MarianMTModel

# Get the name of the model
model_name = 'Helsinki-NLP/opus-mt-en-fr'

# Get the tokenizer
tokenizer = MarianTokenizer.from_pretrained(model_name)
# Instantiate the model
model = MarianMTModel.from_pretrained(model_name)

def format_batch_texts(language_code, batch_texts):
    formated_bach = [">>{}<< {}".format(language_code, text) for text in

                batch_texts]
    return formated_bach

def perform_translation(batch_texts, model, tokenizer, language="fr"):

  # Prepare the text data into appropriate format for the model
  formated_batch_texts = format_batch_texts(language, batch_texts)

  # Generate translation using model
  translated = model.generate(**tokenizer(formated_batch_texts,

                                          return_tensors="pt", padding=True))

  # Convert the generated tokens indices back into text
  translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

  return translated_texts

# Check the model translation from the original language (English) to French
english_texts = news_data['description']
print(english_texts[0])
# translated_texts = perform_translation(english_texts, trans_model, trans_model_tkn)
#
# # Create wrapper to properly format the text
# from textwrap import TextWrapper
# # Wrap text to 80 characters.
# wrapper = TextWrapper(width=80)
#
# for text in translated_texts:
#   print("Original text: \n", text)
#   print("Translation : \n", text)
#   print(print(wrapper.fill(text)))
#   print("")
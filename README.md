# SMSParser
A python program to analyze a message and extract informations like category, amount and due date of paying a bill, message will be ignored if it does not express the sentiment of a bill payment request.

**Requirements**
1. tensorflow==1.6.0
2. numpy==1.14.1
3. pandas==0.22.0
4. gensim==3.5.0
5. nltk==3.3

**Saved Models**
program requires all the models in the savedModels directory for the analysis.
```
1. The directory contains a "bag_of_words.json" file which cosists of list of possible words likely to occur in each category.
2. There's a tensorflow "ner_model" which by analyzing a part of text is cable of determining if the next entity is going to be a "date", "amount" or "other", for example if the context give to the model is "your payment is due on" then it will certaily identify the next token is going to be a date.
3. There's "payment_due_identifier" another tensorflow model which determines if a given message expresses the sentiment of a bill payment request.
There's a word2vec model trained on the given dataset.
```

**Develpment Procedure**
payemnt_due_identifier model is RNN with GRU cells and is developed using supervised training methodologies. Due to absense of labeled data I extracted texts which contains phrases like "your payment is due on", "pay your bill before", "an amount of Rs\s?[0-9]+ is due", this way I had collected engough positive examples and needn't worry about spanning the space of positive class as vectors correspoding to all the keywords and resulting phrases that could be used synonymusly will have higher similarity thus, expanding the horizon of the span. For negative examples I randomly sampled examples from the given dataset and eliminated the once already present in positive examples.

**Usage- Analyzing a dataset**
To analyze collection of messages one must place a "data.csv" file which consists of two columns namely "sender" and "text" in the native directory. A sample of data.csv is already present which contines 10000 random samples.
```
python3 Main.py
```

**Usage- Analyzing a single message**
To extract information from a single message execute the Main.py program with message as an argument.
```
python3 Main.py "message"
```
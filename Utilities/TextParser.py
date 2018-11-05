from .NamedEntityExtractor import NamedEntityExtractor
from .IntentClassifier import IntentClassifier
from .CategoryIdentifier import CategoryIdentifier
import pandas as pd
import numpy as np
import csv

class TextParser:
	def __init__(self):
		self.entity_extractor = NamedEntityExtractor()

	def classify_intents(self, data):
		intent_classifiers = IntentClassifier()
		intents = intent_classifiers.classify(data)
		return intents

	def find_text_category(self, data):
		category_identifier = CategoryIdentifier()
		categories = category_identifier.identify_category_from_data(data)
		return categories 

	def due_date_and_amount(self, text):
		entities = self.entity_extractor.extract_entities(text)
		return entities

	def parse(self, data):
		intents = self.classify_intents(data)
		categories = self.find_text_category(data)
		text_info = list()
		for i, intent in enumerate(intents):
			info = dict({'text': data[i], 'category': categories[i],
						 'amount': None, 'date': None})
			if intent == 'payment_due':
				entity = self.due_date_and_amount(data[i])
				if entity:
					info['amount'] = entity['amount']
					info['date'] = entity['date']
			text_info.append(info)
		return text_info


if __name__ == '__main__':
	parser = TextParser()
	data = pd.read_csv('data.csv').dropna()
	senders, text = data['sender'].values, data['text'].values
	text_info = TextParser().parse(text)
	output = list()
	for i, info in enumerate(text_info):
		if info['amount'] is not None and info['date'] is not None:
			info['sender'] = senders[i]
			output.append(info)

	with open('output.csv', 'w') as f:
		fieldnames = ['sender', 'text', 'category', 'amount', 'date']
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()

		for row in output:
			writer.writerow(row)

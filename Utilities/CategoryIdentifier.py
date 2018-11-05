import re
import json

class CategoryIdentifier:
	def __init__(self):
		self.load_bag_of_words()

	def load_bag_of_words(self):
		with open('savedModels/bag_of_words.json', 'r') as f:
			self.bag_of_words = json.loads(f.read())

	def get_score_template(self):
		score = dict()
		for key in self.bag_of_words.keys():
			score[key] = 0
		return score

	def choose_max_scorer(self, score):
		highest_scorer, highest_score = None, 0
		for key, value in score.items():
			if value > highest_score:
				highest_scorer = key
				highest_score = value
		return highest_scorer

	def is_keyword_in_text(self, keyword, text):
		exp = '[^A-Za-z0-9]' + keyword + '[^A-Za-z0-9]'
		found = re.findall(exp, text, re.IGNORECASE)
		if found:
			return True
		else:
			return False

	def identify_category(self, text):
		score = self.get_score_template()
		bag_of_words = self.bag_of_words
		for key, values in bag_of_words.items():
			for value in values:
				found = self.is_keyword_in_text(value, text) 
				if found:
					score[key] += 1
		category = self.choose_max_scorer(score)
		return category

	def identify_category_from_data(self, data):
		categories = [self.identify_category(s) for s in data]
		return categories
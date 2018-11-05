import tensorflow as tf
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from dateutil.parser import parse
from .Predictor import Predictor
from .Vectorizer import Vectorizer
import re

# returs an amount and a date in context of payment due
class NamedEntityExtractor:
    def __init__(self):
        self.predictor = Predictor('ner_model')
        self.window_size = 5
        self.vectorizer = Vectorizer(self.window_size)
        self.detokenizer = TreebankWordDetokenizer()
        self.prediction_label_map = dict({0: 'amount', 1: 'date', 2: 'none'})

    def identify_price_mentions(self, sentence):
        matches = re.findall('(?:Rs|INR)\s*(?:\,|\.)?\s*[0-9]+(?:\,[0-9]+)*(?:\.[0-9]+)?', sentence)
        matches += re.findall('[0-9]+(?:\,[0-9]+)*(?:\.[0-9]+)?\s*(?:Rs|INR)', sentence)
        return matches


    def identify_date_mentions(self, sentence):
        day = "(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
        month = "(?:january|February|march|april|may|june|July|August|sepetember|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
        matches = re.findall(day, sentence, re.IGNORECASE)
        matches += re.findall('[0-9]{1,4}(?:\-|\/|\.)[0-9]{1,2}(?:\-|\/|\.)[0-9]{1,4}', sentence,
                              re.IGNORECASE)
        matches += re.findall('(?:[0-9]{1,4}(?:\s|\-|\.)?)?' + month +\
                              '(?:(?:\s|\-|\.)?[0-9]{1,4})?',
                              sentence, re.IGNORECASE)
        return matches

    def extract_amount(self, text):
        result = re.findall('[0-9]+(?:\,[0-9]+)?(?:\.[0-9]+)?', text)
        if result:
            return result[0]

    def get_requirements(self, sentence):
        requirements = list()
        words = word_tokenize(sentence)
        for i in range(len(words) + 1):
            if i < self.window_size:
                context = words[:i+1]
            else:
                context = words[i + 1 - 5 : i + 1]
            vector = self.vectorizer.vectorize_words(context)
            vector = np.flip(vector, axis=0)
            info = dict()
            info['context'] = self.detokenizer.detokenize(context)
            info['vector'] = vector
            if i < len(words):
                info['post_context'] = self.detokenizer.detokenize(words[i:])
            else:
                info['post_context'] = ''
            requirements.append(info)
        return requirements

    def parse_date(self, string):
        try:
            date = parse(string)
            year = date.year
            month = date.month
            day = date.day
            date = '%i-%i-%i' % (day, month, year)
            return date
        except Exception as e:
            return None

    def extract_dates(self, text):
        dates = list()
        for i in range(len(text)):
            for j in range(4, 25):
                string = text[i:i+j]
                date = self.parse_date(string)
                if date:
                    dates.append(date)
        dates = list(set(dates))
        return dates

    def predict_entities(self, requirements):
        requirements = list(requirements)
        xdata = np.array([x['vector'] for x in requirements])
        predictions = self.predictor.predict(xdata)
        entities = dict({'date': [], 'amount': []})
        for i, prediction in enumerate(predictions):
            label = self.prediction_label_map[prediction]
            if label == 'amount':
                requirements[i]['label'] = 'amount'
            elif label == 'date':
                requirements[i]['label'] = 'date'
            else:
                requirements[i]['label'] = 'other'
        return requirements

    def entities_exist(self, requirements):
        for elt in requirements:
            if elt['label'] != 'other':
                return True
        return False

    def reformat_date(self, date):
        date_ = self.parse_date(date)
        if date_:
            return date_
        else:
            return date

    def find_date(self, requirements):
        sentence = requirements[0]['post_context']
        known_dates = self.identify_date_mentions(sentence)
        # known_dates = [self.reformat_date(date) for date in known_dates]
        known_dates_formatted = [self.reformat_date(date) for date
                                 in known_dates]
        originals = {known_dates_formatted[i]: known_dates[i] for 
                     i in range(len(known_dates))}
        max_string_size = 25
        for elt in requirements:
            if elt['label'] == 'date':
                string = elt['post_context'][:max_string_size]
                dates = self.extract_dates(string)
                for date in dates:
                    if date in known_dates_formatted:
                        date = originals[date]
                        return date
        if known_dates:
            return known_dates[0]

    def reformat_price(self, price):
        price = re.findall('[0-9]+(?:\,[0-9]+)*(?:\.[0-9]+)?', price)[0]
        return price

    def find_amount(self, requirements):
        known_amounts = self.identify_price_mentions(
            requirements[0]['post_context'])
        known_amounts = [self.reformat_price(price) for price in known_amounts]
        max_string_size = 25
        for elt in requirements:
            if elt['label'] == 'amount':
                string = elt['post_context'][:max_string_size]
                amount = self.extract_amount(string)
                if amount in known_amounts:
                    return amount
        if known_amounts:
            return known_amounts[0]

    def extract_entities(self, sentence):
        requirements = self.get_requirements(sentence)
        requirements = self.predict_entities(requirements)

        # trained ner predicts the presence of date or amount
        # in context of due payment
        date_exists, amount_exists = False, False
        for elt in requirements:
            if not date_exists:
                if elt['label'] == 'date':
                    date_exists = True
            if not amount_exists:
                if elt['label'] == 'amount':
                    amount_exists = True
            if date_exists and amount_exists:
                break

        if not date_exists and not amount_exists:
            return None

        if date_exists:
            date = self.find_date(requirements)
        else:
            date = self.identify_date_mentions(sentence)
            if date:
                date = date[0]
            else:
                return None

        if amount_exists:
            amount = self.find_amount(requirements)
        else:
            amount = self.identify_price_mentions(sentence)
            if amount:
                amount = amount[0]
            else:
                return None

        if date and amount:
            return {'date': date, 'amount': amount}
        else:
            return None
        





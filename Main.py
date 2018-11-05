import sys
from Utilities.TextParser import TextParser
import pandas as pd
import csv

if len(sys.argv) == 1:
	data = pd.read_csv('data.csv').dropna()
	senders, text = data['sender'].values, data['text'].values
	parser = TextParser()
	text_info = parser.parse(text)
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
else:
	message = sys.argv[1]
	parser = TextParser()
	info = parser.parse([message])
	for key, value in info[0].items():
		if key == 'text':
			continue
		print ('=> ', key, ' : ', value)
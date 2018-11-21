import csv
import sys

def remove_javascript(body_text):
	while True:
		firstDelPos=body_text.find("(function")
		secondDelPos=body_text.find("())") 
		if firstDelPos == -1 or secondDelPos == -1:
			break
		body_text = body_text[:firstDelPos] + body_text[secondDelPos+2:]
	return body_text

def remove_unbalanced_parenthesis(body_text):
	to_replace = [")                  ", ")      ","             ", "            ", "      "]
	for r in to_replace:
		body_text = body_text.replace(r, "")
	return body_text

def process_text(f):
	text_file = csv.writer(open('output_clean.csv', 'w'))
	types = set(['columns', 'letters-to-the-editor', 'op-eds', 'editorials'])
	lines = f.readlines()
	for line in lines:
		info = line.split(',')
		type_article, title, link = info[:3]
		body_text = ''.join(info[3:])
		if type_article in types: 
			body_text = remove_javascript(body_text)
			body_text = remove_unbalanced_parenthesis(body_text)
			text_file.writerow([type_article, title, link, body_text])

def main():
	csv.field_size_limit(sys.maxsize)
	f = open('output_all.csv')
	process_text(f)

if __name__ == '__main__':
	main()
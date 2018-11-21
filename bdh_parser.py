from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import time
import csv
"""
TODO:
1. Scrape comments for each article 
2. Record date and time 
3. Record author information
4. Record author position
5. Record article type
"""



def parse(url, div_type):
	req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
	web_byte = urlopen(req).read()
	webpage = web_byte.decode('utf-8')
	soup = BeautifulSoup(webpage, 'html.parser')
	name_box = soup.find('div',attrs={'id': 'main'})
	name_box = name_box.findAll(div_type)
	return name_box	

def parse_article(name_box):
	article_links = []
	for name in name_box:
		link = name.find('a')
		href_link = link['href']
		article_links.append(href_link)
	return article_links

def parse_article_body(url):
	body = ''
	paragraphs = parse(url, 'p')[2:]
	for paragraph in paragraphs:
		body += paragraph.text.replace("\t", "").replace("\r", "").replace("\n", " ")
	return body

def write_article_info(base_url, text_file, article_type):
	name_box = parse(base_url, 'h2')
	article_links = parse_article(name_box)
	assert len(name_box) == len(article_links)
	size = len(article_links)
	for i in range(size):
		name = name_box[i].text.strip()
		link = article_links[i]
		body = parse_article_body(link)
		text_file.writerow([article_type, name, link, body])

def opinions(start, end, base_url, text_file, article_type):
	try:
		write_article_info(base_url, text_file, article_type)
	except:
		pass
	for i in range(start, end):
		new_url = base_url + 'page/' + str(i) + '/'
		try:
			write_article_info(new_url, text_file, article_type)
		except: 
			pass
	
def create_urls():
	url="http://www.browndailyherald.com/sections/opinions/"
	types = ['columns', 'op-eds', 'editorials', 'letters-to-the-editor']
	pages = {'letters-to-the-editor': 51, 'editorials': 77, 'op-eds': 16, 'columns': 211}
	start = 2
	text_file = csv.writer(open('output_all.csv', 'w'))
	for i in types:
		type_url = url+i+'/'
		end = pages[i] + 1
		try:
			opinions(start, end, type_url, text_file, i)
		except:
			pass
	text_file.close()

def main():
	create_urls()

if __name__ == '__main__':
	main()



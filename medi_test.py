from bs4 import BeautifulSoup
from inflection import singularize
import urllib.request, urllib.error, urllib.parse
import re
import os

script_dir         = os.path.dirname(__file__)
names_last         = [l.strip().title() for l in open(os.path.join(script_dir, "medical.txt"))]
en_words           = [l.strip().title() for l in open(os.path.join(script_dir, "en_words.txt"))]

def create_unigrams(input_list):
  bigrams = []
  for i in range(len(input_list)-1):
    bigrams.append((input_list[i].title()))
  return list(set(bigrams))

def main(args):
    # Usage
    if len(args) != 2:
      print("%s <URL or filename>" % args[0])
      return -1

    # Input HTML file or URL
    if os.path.isfile(args[1]):
      html = open(args[1]).read()
    else:
      # Provide a User-Agent
      req = urllib.request.Request(args[1], headers={ 'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; rv:36.0) Gecko/20100101 Firefox/36.0' })
      html = urllib.request.urlopen(req).read()
    
    soup = BeautifulSoup(html, 'html.parser')

    for s in soup(["script", "style"]):
      s.extract()

    particular = " ".join(soup.strings)

    particular = particular.replace("\n", " ")
    regex = re.compile('[^a-zA-Z ]')
    particular = re.sub('\s+', ' ', regex.sub('', particular)).strip()
    particular = [i for i in particular.split() if len(i) > 1]
    particular = ' '.join(particular)

    bigrams = create_unigrams(particular.split())

    indian_names = []
    for name in bigrams:
      if name in names_last and name not in en_words:
        print(name)
      


if __name__ == "__main__":
  import sys 
  sys.exit(main(sys.argv))

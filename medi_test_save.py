from bs4 import BeautifulSoup
from inflection import singularize
import urllib.request, urllib.error, urllib.parse
import re
import os

script_dir         = os.path.dirname(__file__)
names_last         = [l.strip().title() for l in open(os.path.join(script_dir, "medical.txt"))]
names_first_male   = [l.strip().title() for l in open(os.path.join(script_dir, "names.first.male.txt"))]
names_first_female = [l.strip().title() for l in open(os.path.join(script_dir, "names.first.female.txt"))]
names_first_unisex = [l.strip().title() for l in open(os.path.join(script_dir, "names.first.unisex.txt"))]
en_words           = [l.strip().title() for l in open(os.path.join(script_dir, "en_words.txt"))]

def create_bigrams(input_list):
  bigrams = []
  for i in range(len(input_list)-1):
    bigrams.append((input_list[i].title()))
  return list(set(bigrams))

def main(args):
    if len(args) != 2:
      print("%s <URL or filename>" % args[0])
      return -1

    if os.path.isfile(args[1]):
      html = open(args[1]).read()
    else:
      req = urllib.request.Request(args[1], headers={ 'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; rv:36.0) Gecko/20100101 Firefox/36.0' })
      html = urllib.request.urlopen(req).read()
    
    soup = BeautifulSoup(html, 'html.parser')

    for s in soup(["script", "style"]):
      s.extract()

    text = " ".join(soup.strings)

    text = text.replace("\n", " ")
    regex = re.compile('[^a-zA-Z ]')
    text = re.sub('\s+', ' ', regex.sub('', text)).strip()
    text = [i for i in text.split() if len(i) > 1]
    text = ' '.join(text)

    bigrams = create_bigrams(text.split())

    indian_names = []
    for name in bigrams:
      # print(name)
      ln, fn_m, fn_f, fn_u = 0, 0, 0, 0
      en_word, indianness, gender = 0, 0, 0
      if name in names_last and name not in en_words:
        print(name)
        ln = 1
      


if __name__ == "__main__":
  import sys 
  sys.exit(main(sys.argv))

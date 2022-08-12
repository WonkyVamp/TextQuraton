import os
from PIL import Image
from pdf2image import convert_from_path
import pytesseract

filePath = '/home/cockroach/Documents/code_garage/bajaj/venv_baj/test2.pdf'
doc = convert_from_path(filePath)
path, fileName = os.path.split(filePath)
fileBaseName, fileExtension = os.path.splitext(fileName)
text_file = open("/home/cockroach/Documents/code_garage/bajaj/venv_baj/test.txt", "w")
 
 
for page_number, page_data in enumerate(doc):
    txt = pytesseract.image_to_string(page_data).encode("utf-8")
    # print("Page # {} - {}".format(str(page_number),txt))
txt=txt.decode("utf-8")    
text_file.write(txt)
text_file.close()

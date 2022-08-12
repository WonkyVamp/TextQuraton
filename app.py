
# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask,render_template,flash, request, redirect, url_for, jsonify
import urllib.request
import os
from werkzeug.utils import secure_filename
import sys
import random
import json


import json
from inflection import singularize
import urllib.request, urllib.error, urllib.parse
import re
import os

import re
import lexnlp.extract.en.entities.nltk_re
from invoice_gokul_nlp.extract.en.addresses import address_features
import invoice_gokul_nlp.extract.en.dates
import invoice_gokul_nlp.extract.en.money
import nltk
import spacy
import locationtagger 
nltk.downloader.download('maxent_ne_chunker')
nltk.downloader.download('words')
nltk.downloader.download('treebank')
nltk.downloader.download('maxent_treebank_pos_tagger')
nltk.downloader.download('punkt')
nltk.download('averaged_perceptron_tagger')

import requests
#
# from working_test import invoicer
# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)

#files will be uploaded to server in this folder
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 #max 16 mb file upload
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'pdf'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

  
# The route() function of the Flask class is a decorator, 
# which tells the application which URL should call 
# the associated function.

#index
@app.route("/user", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/admin-get-files", methods=["GET"])
#send get request to this endpoint with query parameter as foldername, it will return the model's json response for the corresponding claim file
#example http://localhost:5000/admin-get-files?foldername=71471
def admingetfiles():
    args = request.args
    args = args.to_dict()
    folderName = args.get("foldername")
    with open(os.path.join(app.config['UPLOAD_FOLDER']+folderName,folderName+'.json')) as f:
        data = f.read()
        # return 'in progress'

    res = []
    # Iterate directory
    for path in os.listdir(app.config['UPLOAD_FOLDER']+folderName):
        # check if current path is a file
        if path.split('.')[-1]!='json':
            res.append(path)
    # return res[0]
            # res = path
    return render_template('admin-get-files.html',json_output = data,filename=res)
        # return(str(data[folderName]))



@app.route("/admin", methods=["GET"])
#returns list of all the claim file names that are present in server, can be shown in table and from this table agent should be able to click on the row to send a get request to server, to get the model response file
def admin():
    subfolders = [ f.path for f in os.scandir(app.config['UPLOAD_FOLDER']) if f.is_dir() ]
    # subfolders_text = ''
    # for s in subfolders:
    #     subfolders_text = subfolders_text + s.split('/')[-1] + ' '

    return(render_template('admin.html',subfolders_text=subfolders))

#upload endpoint
@app.route('/upload', methods=['POST'])
def upload_image():
    app.logger.info(request.url)
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filename = str(random.randint(1,10000)) + filename
        # app.logger.info(filename)

        # if filename.rsplit('.', 1)[1].lower() != 'PNG':
        #     images = convert_from_path(os.path.join(app.config['UPLOAD_FOLDER'],filename.split('.')[0],filename))
    
        #     for i in range(1):
        #         # Save pages as images in the pdf
        #         images[i].save(filename.split('.')[0]+'.PNG', 'PNG')

        #     filename = filename.split('.')[0]+'.PNG'

        from pathlib import Path
        Path(app.config['UPLOAD_FOLDER']+filename.split('.')[0]).mkdir(parents=True, exist_ok=True)

        file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename.split('.')[0],filename))
        file.save(os.path.join(app.config['UPLOAD_FOLDER']+filename.split('.')[0],filename.split('.')[0]+'.json'))
        path="/home/cockroach/Documents/code_garage/bajaj/venv_baj/"+os.path.join(app.config['UPLOAD_FOLDER'],filename.split('.')[0],filename)
        json_outs=invoicer(path, path)
        # app.logger.info(path)

        # return(str(json_outs))
        original_stdout = sys.stdout # Save a reference to the original standard output
         
        with open(os.path.join(app.config['UPLOAD_FOLDER'], filename.split('.')[0],filename.split('.')[0]+'.json'), 'w') as f:
            sys.stdout = f # Change the standard output to the file we created.
            print(json_outs) 
            # f.write("jj") 
            # f.write(json_outs)
            sys.stdout = original_stdout # Reset the standard output to its original value


        #print('upload_image filename: ' + filename)
        if filename.rsplit('.', 1)[1].lower() != 'pdf':
            flash('Image successfully uploaded')
            return render_template('index.html')
        else:
            flash('Pdf successfully uploaded to server')
            return redirect(request.referrer)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif, pdf')
        return redirect(request.url)
 
#to display uploaded image in frontend if required  
@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    # return redirect(url_for('static', filename= 'uploads/'+ filename.split('.')[0] + filename), code=301)
    return redirect(url_for('static', filename='uploads/'+filename.split('.')[0]+'/'+filename),code=301)

#sample form to demonstrate form upload to server
@app.route('/info')  
def info():  
   return render_template('info.html')  

#sample form posted to this endpoint
#to demo how data sent to server can be processed and sent back to frontend
@app.route('/success',methods = ['POST', 'GET'])  
def print_data():  
   if request.method == 'POST':  
      result = request.form  
      return render_template("result_data.html",result = result) 
  
def invoicer(inputs, test_in):
    script_dir         = os.path.dirname(__file__)
    names_last         = [l.strip().title() for l in open(os.path.join(script_dir, "names.last.txt"))]
    names_first_male   = [l.strip().title() for l in open(os.path.join(script_dir, "names.first.male.txt"))]
    names_first_female = [l.strip().title() for l in open(os.path.join(script_dir, "names.first.female.txt"))]
    names_first_unisex = [l.strip().title() for l in open(os.path.join(script_dir, "names.first.unisex.txt"))]
    en_words           = [l.strip().title() for l in open(os.path.join(script_dir, "en_words.txt"))]


    medical_names= [l.strip().title() for l in open(os.path.join(script_dir, "medical.txt"))]

    city_names= [l.strip().title() for l in open(os.path.join(script_dir, "cit.txt"))]


    final_dict = {"Extracted Data":[],"Name":[],"Date":[], "City":[], "Pincode":[],"Phone Number":[],"Email id":[], "Possible Service Names":[], "Amounts":[]};



    text_file = open("/home/cockroach/Documents/code_garage/bajaj/venv_baj/test.txt", "w")

    text_file.write(finalstr)
    text_file.close()
    with open('test.txt', 'r') as file:
        info = file.read().rstrip('\n')

    # print("++++")
    # print(info)
    # print("++++")

    final_dict["Extracted Data"].append(info)
    # print(info)
    info=re.sub("\n", " ", info)

    info=info.replace("Rs.", "₹")
    info=info.replace("RS.", "₹")
    info=info.replace("rs.", "₹")
    info=info.replace("rupees", "₹")
    info=info.replace("RUPEES", "₹")
    info=info.replace("Rupees", "₹")
    info=info.replace("Rs", "₹")
    info=info.replace("rs", "₹")
    info=info.replace("RS", "₹")

    # print(info)
    dates_f=list(lexnlp.extract.en.dates.get_dates(info))
    dates_final = re.findall(r'\(.*?\)', str(dates_f))
    dates_f = list(set(dates_f))
    # print(dates_f)
    final_dict["Date"].append(dates_f)
    amounts_final=list(lexnlp.extract.en.money.get_money(info))
    final_dict["Amounts"].append(amounts_final)
    regexp_pin = re.compile(r"([\d])[ -]*?([\d])[ -]*?([\d])[ -]*?([\d])[ -]*?([\d])[ -]*?([\d])")
    regexp_pin = re.compile(r"(\d\d\d\d\d\d\s)")
    # regexp = re.compile(r"\b(\d[\- ]*){6}\b(?<! )")

    # regexp_pin = re.compile(r"^[1-9]{1}[0-9]{2}\s{0,1}[0-9]{3}$")
    pin_c=(re.findall(regexp_pin, info))
    phones_final=[]
    # print(pin_c)
    pin_c_final = list(map("".join, pin_c))
    final_dict["Pincode"].append(pin_c_final)
    # print(list(pin_c_full))
    # regexp_phones= re.compile(r"((\+*)((0[ -]*)*|((91 )*))((\d{12})+|(\d{10})+))|\d{5}([- ]*)\d{6}")

    regexp_phones= re.compile(r"((\+){0,1}91(\s){0,1}(\-){0,1}(\s){0,1}){0,1}[0-9][0-9](\s){0,1}(\-){0,1}(\s){0,1}[1-9]{1}(\s){0,1}(\-){0,1}(\s){0,1}([0-9]{1}(\s){0,1}(\-){0,1}(\s){0,1}){1,6}[0-9]{1}")
    phones=(re.search(regexp_phones, info))
    phones_final=(phones.group())
    # print(phones_final)
    final_dict["Phone Number"].append(phones_final)
    def create_bigrams(input_list):
      bigrams = []
      for i in range(len(input_list)-1):
        bigrams.append((input_list[i].title(), input_list[i+1].title()))
      return list(set(bigrams))
    text=info
    text = text.replace("\n", " ")
    regex = re.compile('[^a-zA-Z ]')
    text = re.sub('\s+', ' ', regex.sub('', text)).strip()

    text = [i for i in text.split() if len(i) > 1]
    text = ' '.join(text)

    bigrams = create_bigrams(text.split())

    indian_names = []
    for name in bigrams:
        ln, fn_m, fn_f, fn_u = 0, 0, 0, 0
        en_word, indianness, gender = 0, 0, 0
        if name[1] in names_last:
            ln = 1
        if name[0] in names_first_male:
            fn_m = 1
        if name[0] in names_first_female:
            fn_f = 1
        if name[0] in names_first_unisex:
            fn_u = 1
      
        indianness = ln + fn_m + fn_f + fn_u
        gender = fn_m - fn_f

        if indianness:
            if name[0] in en_words:
                en_word += 1
        if name[1] in en_words:
            en_word += 1

        singular_fn, singular_ln = singularize(name[0]), singularize(name[1])
        if singular_fn != name[0] and singular_fn in en_words:
            en_word += 1
        if singular_ln != name[1] and singular_ln in en_words:
            en_word += 1

        indianness -= en_word

        if indianness > 0:
            indian_names.append(("%s %s" % name, indianness, gender))

    indian_names = sorted(indian_names, key=lambda x: (-x[1], x[2], x[0]))
    names_final=[]
    for i in indian_names:
        names_final.append(i[0])
    # print(names_final)
    final_dict["Name"].append(names_final)
    def create_unigrams(input_list):
        bigrams = []
        for i in range(len(input_list)-1):
            bigrams.append((input_list[i].title()))
        return list(set(bigrams))

    particular=info
    particular = particular.replace("\n", " ")
    regex = re.compile('[^a-zA-Z ]')
    particular = re.sub('\s+', ' ', regex.sub('', particular)).strip()
    particular = [i for i in particular.split() if len(i) > 1]
    particular = ' '.join(particular)

    bigrams = create_unigrams(particular.split())
    med_terms=[]
    for medical_term in bigrams:
        if medical_term in medical_names and medical_term not in en_words:
            med_terms.append(medical_term)

    final_dict["Possible Service Names"].append(med_terms)



    lst = re.findall('\S+@\S+', info)  

    final_dict["Email id"].append(lst)
    city=info
    city = city.replace("\n", " ")
    regex = re.compile('[^a-zA-Z ]')
    city = re.sub('\s+', ' ', regex.sub('', particular)).strip()
    city = [i for i in city.split() if len(i) > 1]
    city = ' '.join(city)
    bigrams = create_unigrams(city.split())
    final_cities=[]
    for city_find in bigrams:
        if city_find in city_names:
            final_cities.append(city_find)
    final_dict["City"].append(final_cities)      


    jsonString = json.dumps(final_dict, indent=4, default=str)
    return(jsonString)
# main driver function
if __name__ == '__main__':
  
    # run() method of Flask class runs the application 
    # on the local development server.
    app.run(debug=True)

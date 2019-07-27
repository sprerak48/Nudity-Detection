import re, os, keras, random, flask, sys, time, cv2, cgi
import pandas as pd
import numpy as np
import tensorflow as tf
import keras.models
import sqlite3 as sql
import urllib.request
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

sys.path.append(os.path.abspath('./Model'))
from load import *

app = flask.Flask(__name__)  # instantiate flask
app.config["DEBUG"] = True
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

global model, graph
model, graph =  init()

conn = sql.connect('NudeDetect.db')
print ("Opened database successfully")


#Home page
@app.route("/")
def index():
    return render_template("index.html")

#user input
@app.route("/enternewimage")
def enternewimage():
   return render_template("demo.html")


#process in database
@app.route("/addrec",methods = ['POST', 'GET'])
def addrec():
   if request.method == 'POST':
      try:
         nm = request.form['nm']
         url = request.form['image_url']
         #city = request.form['city']
         pic = request.form['image']
         ts = time.time()

         with sql.connect("NudeDetect.db") as con:
            cur = con.cursor()
            cur.execute("INSERT INTO NudeDetectdb (image_name,image_url,image_timestamp,image_data) VALUES (?,?,?,?)",(nm,url,ts,pic))
            con.commit()
            msg = "Record successfully added"
      except:
         con.rollback()
         msg = "error in insert operation"

      finally:
         return render_template("index.html")
         con.close()

#display of table rows
@app.route("/list")
def list():
   con = sql.connect("NudeDetect.db")
   con.row_factory = sql.Row

   cur = con.cursor()
   cur.execute("select * from NudeDetectdb")

   rows = cur.fetchall();
   return render_template("list.html",rows = rows)


#read image and saved
def allowed_image(filename):
    if not "." in filename:
        return False
    ext = filename.rsplit(".", 1)[1]
    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False

@app.route("/upload_image", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            if image.filename == "":
                print("No filename")
                return redirect(request.url)
            if allowed_image(image.filename):
                filename = secure_filename(image.filename)
                image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))
                print("Image saved")
                return redirect(request.url)
            else:
                print("That file extension is not allowed")
                return redirect(request.url)
    return render_template("upload_image.html")


# define a predict function as an endpoint
@app.route("/predict", methods=["GET","POST"])
def predict():
    if request.method == 'POST':
        if request.files:
            target = os.path.join(APP_ROOT, 'Images/')
            print(target)

            if not os.path.isdir(target):
                os.mkdir(target)

            for file in request.files.getlist("image"):
                print(file)
                filename = file.filename
                destination = "/".join([target, filename])
                print(destination)
                file.save(destination)
                #return render_template("index.html")
            x = cv2.imread(destination)
            x = cv2.resize(x,(224,224))
            x = np.reshape(x,[1,224,224,3])
                #convertImage(x)
                #nm = request.form['nm']
            with graph.as_default():
                prediction = model.predict(x).tolist()
                if filename == "":
                    return jsonify(({'status':404, 'image_name' : filename, 'comment' : 'url is not valid' }))
                else:
                    return jsonify(({'status': 200, 'image_name' : filename, 'prediction_value' : prediction}))




if __name__ == "__main__":
    port =  int(os.environ.get('PORT',5000))
    app.run(host='0.0.0.0',port=port,debug=True)

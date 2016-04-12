# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 00:09:49 2016

@author: Kostya S.
"""

from flask import Flask
from flask import render_template
from flask import request, Response , redirect
import json
from model import Model
from collections import OrderedDict

app = Flask(__name__)
global m
m = Model()
m.split_data()
@app.route("/")
def index():
    return render_template('index.html')
    
@app.route('/describe')
def describe():
    return render_template('describe.html')

@app.route("/select_model",methods=['GET', 'POST'])
def select_model():
    global m
    p1 = ''
    m_name = request.form.get('clf')
    p1 = request.form.get('param1')
    tparams = [p1]
    iparams = []
    for e in tparams:
        if e != None:
            iparams.append(int(e))
    
    m = Model(model_name = m_name,params = iparams)
    m.split_data()
    return 'select model is ' + m_name + ' with next parametrs ' + str(p1)

@app.route("/show_train")
def show_train():
    h  = ['sepal length','sepal width','pedal length','pedal width' ,'species']
    return numpy2json(m.get_train_data(),h) 

@app.route("/show_test")
def show_test():
    h  = ['sepal length','sepal width','pedal length','pedal width']
    return numpy2json(m.get_test_data(),h)

@app.route("/train_pred")
def train_pred():
    a = -1
    m.train_pred()
    a = m.get_accuracy()
    return 'Yeeah!!!! Your ' +  'model ' + m.clf.__class__.__name__ + ' was trained with ' + str(a) + ' accuracy!' 
    
@app.route("/show_res")
def show_res():
    h = ['True','Predicted']
    return numpy2json(m.get_class_report(),h)
        
def numpy2json(ds,header = []):
    data = ds
    result = []
    
    for r in zip(data):
        m = min(len(header),len(r[0]))
        rr = r[0][:m]
        row = OrderedDict()
        for h,e in zip(header,rr):
            row[h] = str(e)
        result.append(row)    
    return json.dumps(result)

if __name__ == "__main__":
    app.run()
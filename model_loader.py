from flask import Flask, render_template, request
import tensorflow as tf
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__, static_folder="templates/style.css")

@app.route('/', methods=['GET'])
def helloWorld():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    pdffile = request.files['pdffile']
    pdffile_path = "./pdf_folder/" + pdffile.filename
    pdffile.save(pdffile_path)
    all_files = glob.glob('./pdf_folder/*.csv') #give path to your desired file path
    latest_csv = max(all_files, key=os.path.getctime)
    print (latest_csv)
    data = pd.read_csv(latest_csv)
    my_model = tf.keras.models.load_model('C:/Users/Pranav Joshi/Downloads/my_cnn_model.h5')
    # le = LabelEncoder()
    # mal_encoded = le.fit_transform(data['Malicious'])
    # print(mal_encoded)
    # data['Malicious'] = mal_encoded
    X = data.iloc[:,:21].values
    X_test = np.array(X)
    X_test= X_test.reshape((X_test.shape[0], X_test.shape[1],1))
    y_pred=my_model.predict(X_test)
    y_pred = (y_pred > 0.5)
    print(type(y_pred))
    arr = np.array(y_pred).tolist()
    result_list = []
    final_list=[]
    for i in arr:
        for item in i:
            result_list.append(item)
    for i in range(0, len(result_list)):
        if result_list[i] == False:
            final_list.append('Non malicious')
        else:
            final_list.append('Malicious')
    
    total_files = len(final_list)
    malicious_files = final_list.count('Malicious')
    non_maliciousfiles = final_list.count('Non malicious')

    return render_template('index.html', total_files=total_files, malicious_files=malicious_files, non_maliciousfiles=non_maliciousfiles)


if __name__ == '__main__':
    app.run(port=3000, debug=True)
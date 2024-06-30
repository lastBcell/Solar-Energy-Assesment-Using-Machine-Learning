from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
from geopy.geocoders import Nominatim
import datetime
import xgboost as xgb
import random
# import folium




app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/',methods=['POST','GET'])
def predict():
    data1 = str(request.form['place'])
    geolocator = Nominatim(user_agent="my_app")

# Specify the place name
    place_name = data1

# Get the location information
    location = geolocator.geocode(place_name)

# Extract latitude and longitude
    latitude = location.latitude
    longitude = location.longitude
   
    data2 =int(request.form['months'])
    current_month = datetime.datetime.now().month
    panels = int(request.form['panels'])
    sum=0
    for i in range(0,data2):
        arr = np.array([[latitude,longitude,2020, (current_month+i)%12, 10, 29.5]]).reshape(1,6)
        data=xgb.DMatrix(arr)
        pred = model.predict(data)+ .8+ random.random()
        print(pred)
        sum=sum+pred
    avg=sum/data2
    print(sum,avg)
    sum1=np.squeeze(sum)
    avg1=np.squeeze(avg)
    total_electricty=np.squeeze(panels*(1.5*avg*17*data2*6*30))
    return render_template('pred.html',s=sum1,a=avg1,tl=total_electricty)



if __name__ == '__main__':
    app.run(debug=True)

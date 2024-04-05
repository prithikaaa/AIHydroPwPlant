import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))


@app.route('/')
def index():
    return render_template('hydroHome.html')

@app.route('/explore',methods=['GET','POST'])
def explore():
    features = [
        request.form.get('D.O. (mg/l)'),
        request.form.get('CONDUCTIVITY (µmhos/cm)'),
        request.form.get('B.O.D. (mg/l)'),
        request.form.get('WaterHeadhgt(max)'),
        request.form.get('WaterHeadhgt(min)'),
        request.form.get('Avg Wave Hindcast'),
        request.form.get('Avg Wave Energy Flux'),
        request.form.get('Avg Wave Energy Power')
    ]
  
    # Make predictions
    prediction = model.predict([features])
    #print(prediction)
    return render_template('explore.html', prediction_text=prediction)


if __name__=='__main__':
    app.run(debug=True)
    



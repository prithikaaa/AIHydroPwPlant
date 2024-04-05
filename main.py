import pickle
from flask import Flask, render_template, request

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))


@app.route('/')
def index():
    return render_template('hydroHome.html')

@app.route('/explore',methods=['GET','POST'])
def explore():
    prediction = model.predict([[7.0, 400.0, 2.0009, 200.777, 600.022, 190.0, 8.2, 6.02]])
    print(prediction)
    return render_template('explore.html')


if __name__=='__main__':
    app.run(debug=True)
    



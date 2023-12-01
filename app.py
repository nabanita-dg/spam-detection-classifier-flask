from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)
app.debug=True

# Load the pre-trained machine learning model
model = joblib.load('model/pipe_NaiveBayes.pkl')

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/prediction',methods=['GET','POST'])
def predict():
    mail = request.form['a']
    text=np.array([mail])

    # Perform prediction using the loaded model
    prediction = model.predict(text)

    #return jsonify(prediction.tolist())
    return render_template('result.html', data=int(prediction[0]))

if __name__ == '__main__':
    app.run(port=5000)
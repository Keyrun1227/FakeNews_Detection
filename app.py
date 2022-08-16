from flask import Flask, render_template, request
import pickle

model=pickle.load(open('final_model','rb'))
vector=pickle.load(open('tfidfit_model','rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/home', methods=['POST'])
def home():

    news= request.form['news']
    
    predict=model.predict(vector.transform([news]))[0]
    
    return render_template('home.html',prediction_news='The News Headline Is {}'.format(predict))
    
  
if __name__ == "__main__":
    app.run(debug=True)
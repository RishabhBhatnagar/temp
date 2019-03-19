from flask import render_template, Flask
from ml_nn import predict_single_essay, Constants
import flask

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/submit_essay', methods=["GET", 'POST'])
def submit_essay():
	essay = flask.request.form.get("essay_block")
	print(predict_single_essay(essay, Constants.gensim_model_name, Constants.lstm_model_name))
	return render_template("home.html")

if __name__ == '__main__':
    app.run(debug=1)

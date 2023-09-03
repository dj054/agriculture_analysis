from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor

transformer = pickle.load(open('transformer.sav', 'rb'))
regressor = pickle.load(open('modelforyield.sav', 'rb'))
app = Flask(__name__)



@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        state = str(request.form.get('state'))
        district = str(request.form.get('district'))
        crop = str(request.form.get('crop'))
        season = str(request.form.get('season'))
        year = int(request.form.get('year'))
        production = float(request.form.get('production'))
        area = float(request.form.get('area'))

        data = [[state, district, crop, season, year, production, area]]

        y_test = transformer.transform(data)

        result = float(regressor.predict(y_test))
        return render_template('index.html', pred=result)
    return render_template('index.html', pred='\n')


if __name__ == '__main__':
    app.run(debug=True)

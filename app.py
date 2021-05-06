from flask import Flask, render_template, request
from heartdiseaseprediction import *

app = Flask(__name__)


@app.route('/', methods=["POST", "GET"])
def index():
    if request.method == "POST":
        error = ""
        form = request.form
        age = str(form["age"])
        sex = str(form["sex"])
        chest = str(form["chest"])
        bp = str(form["bp"])
        cholesterol = str(form["cholesterol"])
        fbs = str(form["fbs"])
        ekg = str(form["ekg"])
        hr = str(form["hr"])
        exercise = str(form["exercise"])
        stDepression = str(form["stDepression"])
        stSlope = str(form["stSlope"])
        fluro = str(form["fluro"])
        thallium = str(form["thallium"])
        try:
            if "," in stDepression:
                stDepression = stDepression.replace(",", ".")
            if "," not in stDepression:
                if age.isnumeric() & sex.isnumeric() & chest.isnumeric() & bp.isnumeric() & cholesterol.isnumeric() & fbs.isnumeric() & ekg.isnumeric() & hr.isnumeric() & exercise.isnumeric() & stSlope.isnumeric() & fluro.isnumeric() & thallium.isnumeric():
                    heartdiseasepredict = pipeline.predict(np.array(
                        [[age, sex, chest, bp, cholesterol, fbs, ekg, hr, exercise, stDepression, stSlope, fluro,
                          thallium]]))
                    return render_template('index.html', error="", display=heartdiseasepredict[0])
                else:
                    error = "Please enter a valid value"
                    return render_template('index.html', error=error, display="")
            else:
                error = "Please enter a valid value"
                return render_template('index.html', error=error, display="")
        except:
            error = "Please enter a valid value"
            return render_template('index.html', error=error, display="")
    else:
        return render_template('index.html', display="", error="")


if __name__ == '__main__':
    app.run(debug=False)

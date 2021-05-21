from flask import Flask, render_template, request
import joblib

# __name__ is equal to app.py
app = Flask(__name__)

# load model from model.pck
model = joblib.load('model.h5')
scaler = joblib.load('standard_scaler.h5')



@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')



@app.route("/predict", methods=["POST"])
def predict():

	age =  request.form['Age']
	hypertension = request.form['Hyptertension']
	heart_disease = request.form['Heart Disease']
	avg_glucose_level = request.form['Average Glucose Level']
	bmi = request.form['BMI']
	gender= request.form['Gender']
	ever_married = request.form['Ever Married']
	work_type = request.form['Work Type']
	residence_type = request.form['Residence Type']
	smoking_status = request.form['Smoking Status']

	scaler.transform([[age,avg_glucose_level]])

	if (gender.lower() == 'male'):
		gender_male = 1
	else:
		gender_male = 0

	if (ever_married.lower() == 'yes'):
		ever_married_yes = 1
	else:
		ever_married_yes = 0

	if (work_type.lower() == 'private'):
		work_type_never_worked = 0
		work_type_private = 1
		work_type_self_employed = 0
		work_type_children = 0

	elif (work_type.lower() == 'never_worked'):
		work_type_never_worked = 1
		work_type_private = 0
		work_type_self_employed = 0
		work_type_children = 0

	elif (work_type.lower() == 'self_employed'):
		work_type_never_worked = 0
		work_type_private = 0
		work_type_self_employed = 1
		work_type_children = 0

	elif (work_type.lower() == 'children'):
		work_type_never_worked = 0
		work_type_private = 0
		work_type_self_employed = 0
		work_type_children = 1

	else:
		work_type_never_worked = 0
		work_type_private = 0
		work_type_self_employed = 0
		work_type_children = 0

	if (residence_type.lower() == 'urban'):
		residence_type_urban = 1
	else:
		residence_type_urban = 0

	if (smoking_status.lower() == 'never_smoked'):
		smoking_status_never_smoked = 1
		smoking_status_smokes = 0

	elif (smoking_status.lower() == 'smokes'):
		smoking_status_never_smoked = 0
		smoking_status_smokes = 1

	else:
		smoking_status_never_smoked = 0
		smoking_status_smokes = 0

	# target = predict([[all variables after one hot encoding]])

	int_features = np.array([age, hypertension, heart_disease, avg_glucose_level, bmi, gender_male, ever_married_yes,
							 work_type_never_worked, work_type_private, work_type_self_employed, work_type_children,
							 residence_type_urban, smoking_status_never_smoked, smoking_status_smokes])
	final_features = int_features.reshape(1, -1)

	stroke_state = model.predict(final_features)[0]

	if (stroke_state == 0):
		stroke_diagnosis='You are healthy, good job!'
	else:
		stroke_diagnosis='You aren\'t healthy, please go to nearest doctor'

	return render_template("index.html", stroke_diagnosis=stroke_diagnosis)





if __name__ == "__main__":
    app.run(debug=True)

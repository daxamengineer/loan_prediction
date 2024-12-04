from django.shortcuts import render
import joblib
import numpy as np

# Load models
ann_model = joblib.load('prediction/models/ann.pkl')
logistic_model = joblib.load('prediction/models/Logistic_model.pkl')
random_forest_model = joblib.load('prediction/models/Random_Forest.pkl')

def predict(request):
    if request.method == 'POST':
        # Get input data from the form
        no_of_dependents = float(request.POST.get('no_of_dependents'))
        income_annum = float(request.POST.get('income_annum'))
        loan_amount = float(request.POST.get('loan_amount'))
        cibil_score = float(request.POST.get('cibil_score'))
        residential_assets_value = float(request.POST.get('residential_assets_value'))
        commercial_assets_value = float(request.POST.get('commercial_assets_value'))
        luxury_assets_value = float(request.POST.get('luxury_assets_value'))
        bank_asset_value = float(request.POST.get('bank_asset_value'))
        scaled_loan_term = float(request.POST.get('scaled_loan_term'))

        # Prepare the input for prediction
        input_data = np.array([
            no_of_dependents, income_annum, loan_amount, cibil_score, 
            residential_assets_value, commercial_assets_value, 
            luxury_assets_value, bank_asset_value, scaled_loan_term
        ]).reshape(1, -1)

        print(input_data)
        # Get predictions
        ann_result = ann_model.predict(input_data)[0][0]
        logistic_result = logistic_model.predict([[input_data[0][3],],])[0]
        rf_result = random_forest_model.predict(input_data)[0]

        if ann_result == 1:
            ann_result = 'Eligible'
        else:
            ann_result = 'Not Eligible'

        if rf_result == 1:
            rf_result = 'Eligible'
        else:
            rf_result = 'Not Eligible'

        if logistic_result == 1:
            logistic_result = 'Eligible'
        else:
            logistic_result = 'Not Eligible'           
        

        # Return results to the frontend
        return render(request, 'result.html', {
            'ann_result': ann_result,
            'logistic_result': logistic_result,
            'rf_result': rf_result,
        })

    return render(request, 'predict.html')

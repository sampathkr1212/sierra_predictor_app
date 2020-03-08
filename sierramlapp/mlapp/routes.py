from flask import render_template, url_for, flash, redirect, request, Blueprint, current_app, send_from_directory
import os
import pickle
import pandas as pd
from sierramlapp.mlapp.forms import PricePrediction
from sierramlapp.mlapp.utils import preprocess_data, predict_data, save_file

mlapp = Blueprint('mlapp', __name__)


@mlapp.route('/mlhome',methods=['GET','POST'])
def mlhome():
    form = PricePrediction()
    if form.validate_on_submit():
        if form.excel.data:
            df = pd.read_csv(form.excel.data)
            save_file(form.excel.data)
            global excel_df
            excel_df = df
            dict = df.to_dict(orient='records')
            return render_template('mlhome.html',title='Price Prediction', form=form, dict=dict, len=len, col_names= df.columns)
    return render_template('mlhome.html', title='Price Prediction', form=form)

@mlapp.route('/predict')
def predict():
    form = PricePrediction()
    if excel_df is not None:
        X_data = preprocess_data(excel_df)
        predicted_result = predict_data(X_data)
        excel_df['Price'] = predicted_result
        df = excel_df
        form.excel.data = df.to_csv(index=False)
        data_file_path = os.path.join(current_app.root_path, 'static/data_files')
        result = df.to_csv(data_file_path + '/result.csv',index=False, header=True)
        #save_file(result)
        dict = df.to_dict(orient='records')
        return render_template('mlhome.html',title='Price Prediction', form=form, dict=dict, len=len, col_names= df.columns)
    else:
        flash('Please upload a csv file for prediction', 'info')
        return redirect(url_for('mlapp.mlhome'))

@mlapp.route('/download')
def download():
    file_name = 'result.csv'
    return send_from_directory(filename = file_name, directory = os.path.join(current_app.root_path, 'static/data_files/'), as_attachment=True)

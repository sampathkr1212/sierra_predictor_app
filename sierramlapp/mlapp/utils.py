import os
import secrets
from flask import url_for, current_app
import pandas as pd
import numpy as np
import scipy
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
import pickle
import gc
from werkzeug.utils import secure_filename

count_vectorizer = pickle.load(open('sierramlapp/ml_model_files/count_vectorizer.pkl', 'rb'))
count_category = pickle.load(open('sierramlapp/ml_model_files/count_category.pkl', 'rb'))
count_description = pickle.load(open('sierramlapp/ml_model_files/count_descp.pkl', 'rb'))
vect_brand = pickle.load(open('sierramlapp/ml_model_files/vect_brand.pkl', 'rb'))
ridgeModel = pickle.load(open('sierramlapp/ml_model_files/model.pkl', 'rb'))

def preprocess_data(dataframe):

    dataframe["category_name"] = dataframe["category_name"].fillna("Other").astype("category")
    dataframe["brand_name"] = dataframe["brand_name"].fillna("unknown")
    pop_brands = dataframe["brand_name"].value_counts().index[:2500]
    dataframe.loc[~dataframe["brand_name"].isin(pop_brands), "brand_name"] = "Other"

    dataframe["item_description"] = dataframe["item_description"].fillna("None")
    dataframe["item_condition_id"] = dataframe["item_condition_id"].astype("category")
    dataframe["brand_name"] = dataframe["brand_name"].astype("category")

    X_name = count_vectorizer.transform(dataframe["name"])
    X_category = count_category.transform(dataframe["category_name"])
    X_descp = count_description.transform(dataframe["item_description"])
    X_brand = vect_brand.transform(dataframe["brand_name"])
    X_dummies = scipy.sparse.csr_matrix(pd.get_dummies(dataframe[["item_condition_id", "shipping"]], sparse = True).values,dtype=int)

    X_final_To_Test = scipy.sparse.hstack((X_dummies,
                         X_descp,
                         X_brand,
                         X_category,
                         X_name)).tocsr()
    return X_final_To_Test


def predict_data(X_data):
    preds_log = ridgeModel.predict(X_data)
    predicted_result = np.expm1(preds_log)
    return predicted_result


def save_file(data_file):
    data_file_path = os.path.join(current_app.root_path, 'static/data_files')
    data_file.save(os.path.join(data_file_path, secure_filename(data_file.filename)))

create or replace procedure SVM_Classifier_Train(training_table String)
    returns String
    language python
    runtime_version = 3.8
    packages =(
        'scikit-learn==1.2.2',
        'scipy==1.10.1',
        'snowflake-snowpark-python==*'
    )
    handler = 'main'
    as '# The Snowpark package is required for Python Worksheets. 
# You can add more packages by selecting them using the Packages control and then importing them.

import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col
import pandas as pd
import os
from joblib import dump
import numpy as np
import unicodedata
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.compose import make_column_transformer
from sklearn import metrics
from sklearn import naive_bayes
from sklearn.svm import LinearSVC
from sklearn import ensemble
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
import scipy.sparse as sp
import re
from sklearn import preprocessing 
from sklearn.utils import class_weight
from sklearn.naive_bayes import ComplementNB
import scipy.sparse as sp

def main(session: snowpark.Session, training_table: str)-> str: 
    # Your code goes here, inside the "main" handler.
    df = session.table(training_table).to_pandas()
    df = df[[''MASTER_ITEM_NAME_1'', ''MINOR_SOURCE_NAME'', ''MAJOR_SOURCE_NAME'',  ''PRODUCT_SUB_CATEGORY_NAME'']]
    df = df[df.notnull().all(1)]
    df.rename(columns = {''MASTER_ITEM_NAME_1'': ''MasterItemName'',''MINOR_SOURCE_NAME'': ''MinorSource'', ''MAJOR_SOURCE_NAME'': ''MajorSource'', ''PRODUCT_SUB_CATEGORY_NAME'': ''Category''}, inplace = True)

    def clean_text(text):
        
        # Remove emails 
        text = re.sub(''\\S*@\\S*\\s?'', '''', text)
        
        # Remove new line characters 
        text = re.sub(''\\s+'', '' '', text) 
        
        # Remove distracting single quotes 
        text = re.sub("\\''", '''', text)

        # Remove puntuations and numbers
        text = re.sub(''[^a-zA-Z]'', '' '', text)

        # Remove single characters
        text = re.sub(''\\s+[a-zA-Z]\\s+^I'', '' '', text)
        
        # Remove accented words
        text = unicodedata.normalize(''NFKD'', text).encode(''ascii'', ''ignore'').decode(''utf-8'', ''ignore'')

        # remove multiple spaces
        text = re.sub(r''\\s+'', '' '', text)
        text = re.sub(r''^\\s*|\\s\\s*'', '' '', text).strip()
        text = text.lower()
        return text
    df = df.sample(frac=1).reset_index(drop=True)
    df[''MasterItemName''] = df[''MasterItemName''].apply(lambda x: clean_text(str(x)))
    df[''MinorSource''] = df[''MinorSource''].apply(lambda x: clean_text(str(x)))
    df[''MajorSource''] = df[''MajorSource''].apply(lambda x: clean_text(str(x)))
    df[''Category''] = df[''Category''].apply(lambda x: clean_text(str(x)))
    label_encoder = preprocessing.LabelEncoder()
    df[''Category''] = label_encoder.fit_transform(df[''Category''])
    le_file = os.path.join(''/tmp/'', ''categoryEncoder.joblib'')
    dump(label_encoder, le_file)
    session.file.put(le_file, "@MODELS",overwrite=True)
    x = df.drop(columns = [''Category''], axis = 1)
    y = df[''Category'']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)
    count_vect = CountVectorizer(analyzer="word", token_pattern=r"(?u)\\b\\w\\w+\\b", max_features=10000, ngram_range=(1,1))
    transformer = make_column_transformer((count_vect, ''MasterItemName''), (count_vect, ''MinorSource''), (count_vect, ''MajorSource''))
    transformer.fit(x_train)
    trf_file = os.path.join(''/tmp/'', ''trf.joblib'')
    dump(transformer, trf_file)
    session.file.put(trf_file, "@MODELS",overwrite=True)
    xtrain_count = transformer.transform(x_train)
    xvalid_count = transformer.transform(x_test)
    #x_combined = sp.hstack(x.apply(lambda col: count_vect.fit_transform(col)))
    def train_ml_model(classifier, X_train, X_test, y_train, y_test):

        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        accuracy = metrics.accuracy_score(predictions, y_test)      
        return classifier, accuracy
    
    
    modelSVM, accuracySVM = train_ml_model(LinearSVC(max_iter=15000),
                                        xtrain_count, xvalid_count,
                                        y_train, y_test)
    print("LinearSVC, Count Vectors: ", accuracySVM)
    model_file2 = os.path.join(''/tmp/'', ''SVMmodel.joblib'')
    dump(modelSVM, model_file2)
    session.file.put(model_file2, "@MODELS",overwrite=True)
    print("SVM model saved to stage")

    # Return value will appear in the Results tab.

    return "Model, encoder and Transformer Created and saved"
    ';
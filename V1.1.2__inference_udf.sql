create or replace function categorize_products(MasterItemName string, MinorSource string, MajorSource string)
returns string
language python
runtime_version=3.8
imports=('@sandbox_stage.model_stg/SVMmodel.joblib.gz','@sandbox_stage.model_stg/categoryEncoder.joblib.gz', '@sandbox_stage.model_stg/trf.joblib.gz')
PACKAGES = ('numpy','pandas','joblib', 'scikit-learn==1.2.2', 'regex', 'unicodedata2')
handler='classify'
as
$$
def classify(MasterItemName, MinorSource, MajorSource):
    import os
    import sys
    from joblib import load
    import pandas as pd
    import numpy as np
    import re
    import unicodedata
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,     TfidfTransformer
    from sklearn.compose import make_column_transformer
    IMPORT_DIRECTORY_NAME = "snowflake_import_directory"
    import_dir = sys._xoptions[IMPORT_DIRECTORY_NAME]
    model_name = 'SVMmodel.joblib.gz'
    encoder = 'categoryEncoder.joblib.gz'
    transformer = 'trf.joblib.gz'
    trf = load(import_dir+ transformer)
    le = load(import_dir+ encoder)
    model = load(import_dir+ model_name)
    def clean_text(text):
        
        # Remove emails 
        text = re.sub('\S*@\S*\s?', '', text)
        
        # Remove new line characters 
        text = re.sub('\s+', ' ', text) 
        
        # Remove distracting single quotes 
        text = re.sub("\'", '', text)

        # Remove puntuations and numbers
        text = re.sub('[^a-zA-Z]', ' ', text)

        # Remove single characters
        text = re.sub('\s+[a-zA-Z]\s+^I', ' ', text)
        
        # Remove accented words
        text = unicodedata.normalize('NFKD', text).encode('ascii',       'ignore').decode('utf8', 'ignore')

        # remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'^\s*|\s\s*', ' ', text).strip()
        text = text.lower()
        return text
    predict_data = pd.DataFrame({'MasterItemName': [MasterItemName] , 'MinorSource': [MinorSource], 'MajorSource': [MajorSource]})
    data_vect = trf.transform(predict_data)
    predict_data['MasterItemName'] = predict_data['MasterItemName'].apply(lambda x: clean_text(str(x)))
    predict_data['MinorSource'] = predict_data['MinorSource'].apply(lambda x: clean_text(str(x)))
    predict_data['MajorSource'] = predict_data['MajorSource'].apply(lambda x: clean_text(str(x)))    
    scored_data = model.predict(data_vect)[0]
    scores = np.atleast_1d(scored_data)
    predicted_categories = le.inverse_transform(scores)

    return predicted_categories[0]
    
$$;

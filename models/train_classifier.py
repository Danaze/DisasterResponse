# import necessary libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet'])
import pandas as pd
from sqlalchemy import create_engine
import sqlite3
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
import pickle

def load_data(database_filepath):
    '''
    Parameters
    ----------
    database_filepath : path for the database

    Returns
    -------
    A dataset for predicted and predictor variables (y and X) and list of target categories.

    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)
    X = df.loc[:, 'message']
    y = df.iloc[:, 5:]
    categories = y.columns
    return X, y, categories
def tokenize(text):
    '''
    A function to tokenize a text input

    Parameters
    ----------
    text : input text 

    Returns
    -------
    clean_tokens : list of tokens created from the input text
    '''
    # tokenize text
    tokens = word_tokenize(text)
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    # iterate through each token
    clean_tokens = []
    for tok in tokens:  
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model(grid_search = False):
    '''
    A function that returns a pipeline of svc multioutput classfier to be used for model fitting.
    Parameter: grid_search (bool) controls whether to do a grid search cross validation for the model parameters
    Returns: a multioutput classifier model pipeline
    '''

    svc = LinearSVC(random_state=0)
    multi_class_svc = OneVsRestClassifier(svc)
    
    pipeline =  Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf',MultiOutputClassifier(multi_class_svc))
        ])
    if grid_search:
        print("grid searching for best parameters.")
        parameters = {
            'clf__estimator__estimator__C': [0.1, 0.5, 1, 10],
            'clf__estimator__estimator__max_iter': [100, 500, 1000, 5000],
            }

        pipeline = GridSearchCV(pipeline, param_grid=parameters)
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    A function to evaluate the model
    Parameters: model, test data and target categories
    The function prints precision, recall, f1score and accuracy for each categories as well as overall metrics.
    '''
    y_pred = model.predict(X_test)
    accuracy = []
    precision_list = []
    recall_list = []
    f1_score_list = []
    for idx, col in enumerate(category_names):
        print('Category: {} '.format(category_names[idx]))
        print(classification_report(Y_test.iloc[:,idx], y_pred[:,idx]))
        metric = precision_recall_fscore_support(Y_test.iloc[:,idx], y_pred[:,idx])
        precision = 0
        recall = 0
        f1score = 0
        for i in range (len(metric[0])):
            precision = precision + metric[0][i] * metric[3][i]
            recall = recall + metric[1][i] * metric[3][i]
            f1score = f1score + metric[2][i] * metric[3][i]
        avg_precision = precision /(sum(metric[3]))
        avg_recall = recall /(sum(metric[3]))
        avg_f1score = f1score /(sum(metric[3]))
        precision_list.append(avg_precision)
        recall_list.append(avg_recall)
        f1_score_list.append(avg_f1score)
        print('avg precision: ', avg_precision)
        print('avg recall: ', avg_recall)
        print('avg f1score: ', avg_f1score)
        print('Accuracy {}\n\n'.format(accuracy_score(Y_test.iloc[:, idx].values, y_pred[:, idx])))
        accuracy.append(accuracy_score(Y_test.iloc[:, idx].values, y_pred[:, idx]))
    print('Overall average accuracy: {} '.format(sum(accuracy)/len(accuracy)))
    print('Overall average precision: {} '.format(sum(precision_list)/len(precision_list)))
    print('Overall average recall: {} '.format(sum(recall_list)/len(recall_list)))
    print('Overall average f1score: {} '.format(sum(f1_score_list)/len(f1_score_list)))


def save_model(model, model_filepath):
    '''
    A function to save the model in pickle format.

    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
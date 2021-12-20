import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    A function to load the original messages and categories data and merge them in a dataframe.
    
    Parameters:
        messages_filepath: filepath for messages data
        categories_filepath: filepath for categories data
    Returns:
        a dataframe merging messages and categories data
    '''
    # load messages
    messages = pd.read_csv(messages_filepath, dtype = str)
    # load categories
    categories = pd.read_csv(categories_filepath, dtype = str)
    # merge categories and messages
    df = pd.merge(categories, messages, on="id")
    return df
def clean_data(df):
    '''
    A function to clean the dataframe
    
    Parameters:
        df: the dataframe returned from loading data (merged messages and categories)
    Returns:
        a dataframe cleaned, i.e., categories expanded and duplicates removed.
    '''
    # expand the categories to add column for each category type
    categories = df['categories'].str.split(';', expand=True)
    # create a list of categories from the first row
    row = categories.iloc[0].tolist()
    # slice the rows upto the second to last character of each string
    category_colnames = [x[:-2] for x in row]
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        categories[column] = (categories[column] > 0).astype(int)
    # drop the original categories column
    df = df.drop(['categories'], axis=1)
    # concatenate the df with categories dataframe created above
    df = pd.concat([df, categories], axis=1)
    # drop duplicate rows
    df = df.drop_duplicates()
    return df
def save_data(df, database_filename):
    '''
    A funciton for saving the cleaned dataframe into a database with name given as a parameter.
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages', engine, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
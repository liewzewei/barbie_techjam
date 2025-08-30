import os
import pandas as pd

def convert_json_to_csv(json_file_path,sample_size):
    """
    Reads data from a JSON file, converts it to a pandas DataFrame,
    and saves it as a CSV file. Note that the JSON file must have a 
    'text' column.

    This function is robust enough to handle two common JSON formats:
    1. A JSON array of objects: [{}, {}, {}]
    2. A JSON lines file, where each line is a JSON object: {}\n{}\n{}

    This function isolates the 'text' column of the table and cleans
    it by removing empty columns and special non-alphanumeric characters.

    Args:
        json_file_path (str): The path to the input JSON file.
        sample_size (int): Number of random samples in output csv
    """
    try:
        # The most common format for large datasets is JSON lines.
        # We'll try this first as it's very memory-efficient.
        print("\nAttempting to read JSON file as JSON lines format...")
        df = pd.read_json(json_file_path, lines=True)
        print("Successfully read JSON lines file.")

    except ValueError:
        # If the above fails, it might be a standard JSON array.
        print("JSON lines failed. Attempting to read as standard JSON array...")
        # For a standard JSON array, you can just use read_json without lines=True
        df = pd.read_json(json_file_path)
        print("Successfully read standard JSON file.")

    except Exception as e:
        print(f"An error occurred: {e}")
        return

    print('Cleaning ...')
    df_single = df[['text']]
    df_single = df_single.dropna()
    random_sample = df_single.sample(n=sample_size)
    regex_str = r'[^a-zA-Z0-9 ]'
    random_sample['cleaned_text'] = random_sample['text'].str.replace(regex_str,'',regex=True)
    cleaned = random_sample.drop(columns=['text'])
    print('Cleaning completed')
    print('\nSaving cleaned csv')
    json_name = os.path.basename(json_file_path)
    file_name = os.path.splitext(json_name)[0]
    csv_name = 'Cleaned_' + file_name + '.csv'
    csv_path = os.path.join(os.path.dirname(json_file_path),csv_name)
    cleaned.to_csv(csv_path,index=False)
    print('Saved cleaned csv')

'''
Enter the path of your json below

'''

path = ''

print('Processing ...')
convert_json_to_csv(path,20000)
print('Completed')
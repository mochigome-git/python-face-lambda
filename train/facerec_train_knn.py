"""
k-nearest-neighbors (KNN) algorithm for face recognition.

Algorithm Description:
The knn classifier is first trained on a set of labeled (known) faces and can then predict the person
in an unknown image by finding the k most similar faces (images with closet face-features under euclidean distance)
in its training set, and performing a majority vote (possibly weighted) on their label.

For example, if k=3, and the three closest face images to the given image in the training set are one image of Biden
and two images of Obama, The result would be 'Obama'.

* This implementation uses a weighted vote, such that the votes of closer-neighbors are weighted more heavily.

NOTE: This example requires scikit-learn to be installed! You can install it with pip:

$ pip3 install scikit-learn

"""

import math
from sklearn import neighbors
import os
import os.path
import pickle
import requests
import numpy as np
import boto3

#from dotenv import load_dotenv
#import face_recognition
#from face_recognition.face_recognition_cli import image_files_in_folder

# Local enviroment test
#load_dotenv()
#SUPABASE_URL = os.getenv('SUPABASE_URL')
#SUPABASE_KEY = os.getenv('SUPABASE_KEY')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def get_parameter_from_ssm(name):
    ssm = boto3.client('ssm')
    try:
        parameter = ssm.get_parameter(Name=name, WithDecryption=True)
        return parameter['Parameter']['Value']
    except ssm.exceptions.ParameterNotFound:
        print(f"Parameter {name} not found in SSM Parameter Store.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def fetch_data_from_database():
    # Access environment variables from Lambda runtime
    SUPABASE_URL = os.environ.get('SUPABASE_URL') 
    SUPABASE_KEY = os.environ.get('SUPABASE_KEY')

    # Ensure SUPABASE_KEY is retrieved from SSM parameter store
    DE_SUPABASE_KEY = get_parameter_from_ssm(SUPABASE_KEY)

    headers = {
        "apikey": DE_SUPABASE_KEY,
        "Authorization": f"Bearer {DE_SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    response = requests.get(SUPABASE_URL, headers=headers)

    if response.status_code == 200:
            data = response.json()
            names = []
            encodings = []
            ids = []
            for record in data:
                names.append(record['name'])
                ids.append(record['face_id'])
                encodings.append(np.array(record['face_encoding']))
            return names, encodings, ids
    else:
        print("Failed to fetch data from the database. Status code:", response.status_code)
        return [], []

def train(model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    Trains a k-nearest neighbors classifier for face recognition.
    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    names, encodings, ids = fetch_data_from_database()

    # Check if data is fetched successfully
    if not names or not encodings or not ids:
        return None

    # Convert list of arrays to numpy array
    X = np.array(encodings)
    y = np.array(ids)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf

def lambda_handler(event, context):
    # Assuming event contains any data you want to pass to the Lambda function
    # You can modify this function to handle specific event data
    try:
        print("Received event:", event)
        
        # Perform necessary operations here
        # Example: train the classifier
        classifier = train(n_neighbors=2)
        
        if classifier:
            print("Training complete!")
            
            # Serialize the trained model
            serialized_model = pickle.dumps(classifier)
            
            # Upload the serialized model to S3
            bucket_name = os.environ.get('BUCKET_NAME')
            key = os.environ.get('KEY_NAME')  # Specify the key (filename) under which to store the model in S3
            s3_client = boto3.client('s3')
            s3_client.put_object(Bucket=bucket_name, Key=key, Body=serialized_model)
            
            print("Model uploaded to S3 successfully!")
            return {
                'statusCode': 200,
                'body': 'Classifier trained and model uploaded to S3 successfully!'
            }
        else:
            print("Failed to train classifier.")
            return {
                'statusCode': 500,
                'body': 'Failed to train classifier.'
            }
        
    except Exception as e:
        # Handle any unexpected exceptions
        print("An error occurred:", str(e))
        return {
            'statusCode': 500,
            'body': f"An error occurred: {str(e)}"
        }

# Uncomment below lines if you want to test the lambda_handler locally
# event = {}  # example event
# context = {}  # example context
# print(lambda_handler(event, context))
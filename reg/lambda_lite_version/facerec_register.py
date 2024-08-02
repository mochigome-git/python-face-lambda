import json
import os
import requests
import numpy as np
import boto3
import face_recognition
import tempfile
from urllib.parse import urlparse


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
    
def get_photo_from_s3(url):
    parsed_url = urlparse(url)
    bucket_name = parsed_url.netloc.split('.', 1)[0]  # Extract bucket name without 's3.hostname'
    key = parsed_url.path.lstrip('/')
    s3 = boto3.client('s3')
    try:
        response = s3.get_object(Bucket=bucket_name, Key=key)
        image = response['Body'].read()
        return image
    except Exception as e:
        print(f"Error getting object {key} from bucket {bucket_name}: {e}")
        return None

def recognize_faces_in_image(name_pic, face_id, image_path=None, image_encoded_array=None):
    """
    Recognizes faces in the given image and stores face encodings in Supabase.

    Args:
    :param image_path (optional): Path to the image file.
    :param image_encoded_array (array): Encoded image array.
    :param name_pic (str): Image owner's name.
    :param face_id (int): id of the owner registered in facedb.
    
    Returns:
    :return: Tuple containing the result and debug information.
    """    
    # Access environment variables from Lambda runtime
    SUPABASE_URL = os.environ.get('SUPABASE_URL') 
    SUPABASE_KEY = os.environ.get('SUPABASE_KEY')

    # Ensure SUPABASE_KEY is retrieved from SSM parameter store
    DE_SUPABASE_KEY = get_parameter_from_ssm(SUPABASE_KEY)

    if not DE_SUPABASE_KEY:
        print("Error: SUPABASE_KEY could not be retrieved from SSM.")
        return "Error: KEY not retrieved correctly from SSM."
    
    if image_path is None and image_encoded_array is None:
        raise ValueError("Must supply face recognizer register either through image_path or image_encoded_array")

    try:
        # Case when using image_path
        if image_encoded_array is None:
            image_data = get_photo_from_s3(image_path)
            if image_data:
                # Save image data to a temporary file
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(image_data)
                    tmp_file_path = tmp_file.name
                    print(f"Temporary file path: {tmp_file_path}")

                # Load image from the temporary file for face recognition
                image = face_recognition.load_image_file(tmp_file_path)

                # Find all face locations and encodings
                face_locations = face_recognition.face_locations(image)
                face_encodings = face_recognition.face_encodings(image, face_locations)
                # Display face encodings
                for face_encoding in face_encodings:
                    json_encoded = face_encoding.tolist()

                # Clean up: remove temporary file
                os.remove(tmp_file_path)
        else:
            print("Failed to retrieve image data from S3.")

        # Case when receive encoded data
        ## Store encoded image data directly 
        if image_path is None:
            face_encoding = np.array(image_encoded_array)
            json_encoded = face_encoding.tolist()

        # Store face encodings in Supabase
        headers = {
            "apikey": DE_SUPABASE_KEY,
            "Authorization": f"Bearer {DE_SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }
        data = {
            "name": name_pic,
            "face_encoding": json_encoded,
            "face_id": face_id
        }

        response = requests.post(SUPABASE_URL, headers=headers, json=data)
        
        # Check response status
        if response.status_code == 201:
            result = "Face encodings stored successfully."
        else:
            result = f"Failed to store face encodings. Status code: {response.status_code, response.text}"

    except Exception as e:
        result = f"Error: {e}"

    return result

def lambda_handler(event, context):
    lambda_client = boto3.client('lambda')

    SUCCESS_FUNCTION_ARN = os.environ['SUCCESS_FUNCTION_ARN']
    FAILURE_FUNCTION_ARN = os.environ['FAILURE_FUNCTION_ARN']

    try:
        # Parse incoming JSON payload
        body = json.loads(event['body'])
        name_pic = body.get('name_pic')
        face_id = body.get('face_id')
        image_encoded_array = body.get('image_encoded_array')
        image_path = body.get('image_path')

        # Call processing function with the provided data
        result = recognize_faces_in_image(name_pic, face_id, image_path=image_path, image_encoded_array=image_encoded_array)

        # On success, invoke the success Lambda function
        response_payload = json.dumps({"response": result})
        lambda_client.invoke(
            FunctionName=SUCCESS_FUNCTION_ARN,
            InvocationType='Event',  # Asynchronous invocation
            Payload=response_payload
        )

        return {
            "statusCode": 201,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({"response": result})
        }
    
    except Exception as e:

        # On failure, invoke the failure Lambda function
        error_payload = json.dumps({"error": str(e)})
        lambda_client.invoke(
            FunctionName=FAILURE_FUNCTION_ARN,
            InvocationType='Event',  # Asynchronous invocation
            Payload=error_payload
        )

        # Print exception for debugging
        print("Exception:", e)
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({"error": str(e)})
        }


import face_recognition
import numpy as np
import requests
import json
import os

# Constants for local test
'''
load_dotenv()
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
'''
# Access environment variables from Lambda runtime
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')

def recognize_faces_in_image(name_pic, user_id, image_path=None, image_encoded_array=None):
    """
    Recognizes faces in the given image and stores face encodings in Supabase.

    Args:
    :param image_path (optional): Path to the image file.
    :param image_encoded_array (array): Encoded image array.
    :param name_pic (str): Image owner's name.
    :param user_id (int): id of the owner registered in facedb.
    
    Returns:
    :return: Tuple containing the result and debug information.
    """
    if image_path is None and image_encoded_array is None:
        raise ValueError("Must supply face recognizer register either through image_path or image_encoded_array")
    
    json_encoded = None  # Define json_encoded outside the conditional blocks

    try:
        # Case when using image_path
        if image_encoded_array is None:
            image = face_recognition.load_image_file(image_path)
            # Find all face locations and encodings
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            # Display face encodings
            for face_encoding in face_encodings:
                json_encoded = face_encoding.tolist()  # Corrected indentation

        # Case when receive encoded data
        # Store encoded image data directly 
        if image_path is None:
            face_encoding = np.array(image_encoded_array)
            json_encoded = face_encoding.tolist()

        # Store face encodings in Supabase
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }
        data = {
            "name": name_pic,
            "face_encoding": json_encoded,
            "user_id" : user_id
        }

        response = requests.post(SUPABASE_URL, headers=headers, json=data)
        
        # Check response status
        if response.status_code == 201:
            result = "Face encodings stored successfully."
        else:
            result = f"Failed to store face encodings. Status code: {response.status_code}"
        
    except FileNotFoundError as e:
        result = "Image file not found."
    except Exception as e:
        result = f"Error: {e}"

    return result


def lambda_handler(event, context):
    # Parse incoming JSON payload
    try:
        body = json.loads(event['body'])
        name_pic = body.get('name_pic')
        user_id = body.get('user_id')
        image_encoded_array = body.get('image_encoded_array')
        image_path = body.get('image_path')
        
        # Call processing function with the provided data
        result = recognize_faces_in_image(name_pic, user_id, image_path=image_path, image_encoded_array=image_encoded_array)
        message = result
        
        return {
            "statusCode": 201,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({"response": message})
        }
    except Exception as e:
        # Print exception for debugging
        print("Exception:", e)
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({"error": str(e)})
        }
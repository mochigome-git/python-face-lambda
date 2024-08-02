import io
import cv2
import base64
import pickle
import face_recognition
import numpy as np
import boto3
import os
from PIL import Image
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
from concurrent.futures import ThreadPoolExecutor
# Uncomment import and calls when create docker image
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
executor = ThreadPoolExecutor(max_workers=16)

ACCESS_ID = os.environ.get('ACCESS_ID')
ACCESS_KEY = os.environ.get('ACCESS_KEY')
BUCKET_NAME = os.environ.get('BUCKET_NAME')
OBJECT_PATH = os.environ.get('OBJECT_PATH')

session = boto3.Session(
    aws_access_key_id= ACCESS_ID,
    aws_secret_access_key= ACCESS_KEY,
)

def load_model_from_s3(bucket_name, object_path):
    s3 = session.resource('s3')
    obj = s3.Bucket(bucket_name).Object(object_path)
    knn_model = pickle.loads(obj.get()['Body'].read())
    return knn_model

# Define Obejct Bucket path stored in S3
bucket_name= BUCKET_NAME
object_path= OBJECT_PATH
# Load a trained KNN model 
knn_clf = load_model_from_s3(bucket_name, object_path)

def predict_faces(frame, resize_factor=1):
    # Resize the frame for faster processing
    frame = cv2.resize(frame, (int(frame.shape[1] * resize_factor), int(frame.shape[0] * resize_factor)))
    # Predict faces in the frame
    predictions = []
    # Find face locations
    face_box = face_recognition.face_locations(frame)
    print(f"Detected {len(face_box)} faces") 
    # If no faces are found in the image, return an empty result.
    if len(face_box) == 0:
        return []
    # Find encodings for faces in the test image
    faces_encodings = face_recognition.face_encodings(frame, known_face_locations=face_box)
    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=2)
    threshold = 0.4
    matches = [closest_distances[0][i][0] <= threshold for i in range(len(face_box))]
    
    predictions = [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), face_box, matches)]
    print(f"Predictions: {predictions}") 
    # Predict classes and remove classifications that aren't within the threshold
    return predictions

# Define a function to encode the frame in WEBP format
def encode_frame(resized_frame):
    try:
        _, buffer = cv2.imencode('.webp', resized_frame)
        stringData = base64.b64encode(buffer).decode('utf-8')
        b64_src = 'data:image/webp;base64,'
        stringData = b64_src + stringData
        return stringData
    except Exception as e:
        print("Error encoding frame:", e)
        return None
    
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@socketio.on('image')
def image(data_image):
    # Frame counter (initialize to 0)
    frame_count = 0
   
    # Retry logic for drawing the frame and encoding
    max_retry = 10
    retry_count = 5
    while retry_count < max_retry:
        try:
            # Decode and convert into image
            b = io.BytesIO(base64.b64decode(data_image))
            pimg = Image.open(b)

            # Converting RGB to BGR, as OpenCV standards
            frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

            # Resize the frame (adjust resize_factor as needed)
            resize_factor = 0.3
            resized_frame = cv2.resize(frame, (int(frame.shape[1] * resize_factor), int(frame.shape[0] * resize_factor)))

            # Predict faces in the frame
            future = executor.submit(predict_faces, resized_frame)
            # Process the frame only if not skipped (original logic)
            frame_count += 1
            # Draw boxes and names on the frame
            predictions = future.result()
            for name, (top, right, bottom, left) in predictions:
                cv2.rectangle(resized_frame, (left, top), (right, bottom), (0, 255, 255), 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                
                # Adjust the text position
                text_x = left - 5
                text_y = top - 3
                
                cv2.putText(resized_frame, name, (text_x, text_y), font, 0.8, (255, 255, 255), 1)
            
            # Encode the frame in WEBP format
            stringData = encode_frame(resized_frame)
            # Emit the frame back if encoding is successful
            if stringData is not None:
                emit('response_back', stringData)
                break
            else:
                # If encoding failed, increment retry count and retry
                retry_count += 1
                print("Retry", retry_count)
            
        except Exception as e:
            print(f"Error processing image: {e}")
            # If any exception occurs, increment retry count and retry
            retry_count += 1
            print("Retry", retry_count)


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=5000)

# usage simple : gunicorn -w 4 -b 0.0.0.0:5000 --worker-class eventlet facerec_rec_known_shface:app
# usage : gunicorn -w 4 -b 0.0.0.0:5000 --worker-class eventlet facerec_rec_known:app & gunicorn --certfile=certificates/server.crt --keyfile=certificates/server.key -w 4 -b 0.0.0.0:5080 --worker-class eventlet facerec_rec_known_shface:app
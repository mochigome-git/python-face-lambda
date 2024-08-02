import io
import cv2
import base64
import pickle
import face_recognition
import numpy as np
import boto3
import os
import time
from PIL import Image
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
from concurrent.futures import ThreadPoolExecutor
from flask_cors import CORS
from dotenv import load_dotenv
from threading import Thread, Event, Lock

# Load environment variables from .env file
# load_dotenv()
# 
# ACCESS_ID = os.getenv('ACCESS_ID')
# ACCESS_KEY = os.getenv('ACCESS_KEY')
# BUCKET_NAME = os.getenv('BUCKET_NAME')
# OBJECT_PATH = os.getenv('OBJECT_PATH')
# DOMAIN_ORIGIN = os.getenv('DOMAIN_ORIGIN')

ACCESS_ID = os.environ.get('ACCESS_ID')
ACCESS_KEY = os.environ.get('ACCESS_KEY')
BUCKET_NAME = os.environ.get('BUCKET_NAME')
OBJECT_PATH = os.environ.get('OBJECT_PATH')
DOMAIN_ORIGIN = os.environ.get('DOMAIN_ORIGIN')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": [DOMAIN_ORIGIN]}})
socketio = SocketIO(app, async_handlers=True, allow_upgrades=False, cors_allowed_origins=DOMAIN_ORIGIN)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['SECRET_KEY'] = 'secret!'
executor = ThreadPoolExecutor(max_workers=16)

session = boto3.Session(
    aws_access_key_id= ACCESS_ID,
    aws_secret_access_key= ACCESS_KEY,
)

def load_model_from_s3(bucket_name, object_path):
    s3 = session.resource('s3')
    obj = s3.Bucket(bucket_name).Object(object_path)
    knn_model = pickle.loads(obj.get()['Body'].read())
    return knn_model

def get_last_modified_time(bucket_name, object_path):
    s3 = session.resource('s3')
    obj = s3.Bucket(bucket_name).Object(object_path)
    return obj.last_modified

def check_for_updates(interval, stop_event):
    global knn_clf
    global knn_lock
    last_modified_time = get_last_modified_time(BUCKET_NAME, OBJECT_PATH)
    while not stop_event.is_set():
        time.sleep(interval)
        current_modified_time = get_last_modified_time(BUCKET_NAME, OBJECT_PATH)
        if current_modified_time != last_modified_time:
            print("KNN model updated, reloading...")
            new_knn_clf = load_model_from_s3(BUCKET_NAME, OBJECT_PATH)
            with knn_lock:
                knn_clf = new_knn_clf
            last_modified_time = current_modified_time

# Define Obejct Bucket path stored in S3
bucket_name= BUCKET_NAME
object_path= OBJECT_PATH

# Load a trained KNN model
knn_clf = load_model_from_s3(bucket_name, object_path)
knn_lock = Lock()

# Initialize a variable to store the previous face count
previous_face_count = None

# Create an event to stop the background thread
stop_event = Event()
# Start the background thread to check for updates
update_thread = Thread(target=check_for_updates, args=(5, stop_event))  # Check every 60 seconds
update_thread.start()

def predict_faces(frame, resize_factor=1):
     # Use a global variable to track previous face count and knn file
    global previous_face_count
    global knn_clf
    global knn_lock
    
    # Resize the frame for faster processing
    frame = cv2.resize(frame, (int(frame.shape[1] * resize_factor), int(frame.shape[0] * resize_factor)))
    # Predict faces in the frame
    predictions = []
    # Find face locations
    face_box = face_recognition.face_locations(frame)

    # Number of detected faces in the current frame
    current_face_count = len(face_box) 
    # Print detection message if the number of faces has changed or if it's the first detection
    if current_face_count != previous_face_count:
        print(f"Detected {current_face_count} faces")
        previous_face_count = current_face_count  # Update previous count

    # If no faces are found in the image, return an empty result.
    if len(face_box) == 0:
        return []
    # Find encodings for faces in the test image
    faces_encodings = face_recognition.face_encodings(frame, known_face_locations=face_box)
    # Use the KNN model to find the best matches for the test face
    with knn_lock:
        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=2)
        threshold = 0.5
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

@socketio.on('connect')
def handle_connect():
    try:
        print('Client connected')
        # Your handling logic here
    except Exception as e:
        print(f"Error in handle_connect: {e}")
        # Handle the error appropriately, e.g., emit an error message or log it

@socketio.on('disconnect')
def handle_disconnect():
    try:
        print('Client disconnected')
        # Your handling logic here
    except Exception as e:
        print(f"Error in handle_disconnect: {e}")
        # Handle the error appropriately

@socketio.on('image')
def image(data_image):
    # Retry logic for drawing the frame and encoding
    max_retry = 10
    retry_count = 5
    target_fps = 24
    frame_duration = 3.0 / target_fps 

    while retry_count < max_retry:
        start_time = time.time()
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

            # Draw boxes and names on the frame
            predictions = future.result()
            for name, (top, right, bottom, left) in predictions:
                cv2.rectangle(resized_frame, (left, top), (right, bottom), (0, 255, 255), 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                text_position = (left, top - 10 if top - 10 > 10 else top + 10)
                cv2.putText(resized_frame, name, text_position, font, 1, (255, 255, 255), 1)

            # Get the face owner's name if available
            owner_name = predictions[0][0] if predictions else 'unknown'

            # Encode the frame in WEBP format
            stringData = encode_frame(resized_frame)
            if stringData is not None:
                # Emit the annotated image and face owner's name back to the frontend
                emit('response_back', {'image': stringData, 'name': owner_name})
                break
            else:
                # If encoding failed, increment retry count and retry
                raise ValueError("Failed to encode frame.")
            
        except Exception as e:
            print("Retry {}/{}: Error processing image:".format(retry_count, max_retry), e)
            retry_count += 1  # Increment retry count on failure

        # Ensure the loop runs at the target frame rate
        elapsed_time = time.time() - start_time
        sleep_time = frame_duration - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)

# @app.route('/')
# def index():
#  return render_template('index.html')

if __name__ == "__main__":
    try:
        socketio.run(app, host='0.0.0.0', port=5000)
    finally:
        stop_event.set()
        update_thread.join()

# usage simple : gunicorn -w 4 -b 0.0.0.0:5000 --worker-class eventlet facerec_rec_known_shface:app
# usage : gunicorn -w 4 -b 0.0.0.0:5000 --worker-class eventlet facerec_rec_known:app & gunicorn --certfile=certificates/server.crt --keyfile=certificates/server.key -w 4 -b 0.0.0.0:5080 --worker-class eventlet facerec_rec_known_shface:app
import io
import cv2
import base64
import pickle
import face_recognition
import numpy as np
import boto3
from PIL import Image
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
executor = ThreadPoolExecutor(max_workers=8)

session = boto3.Session(
    aws_access_key_id="aws_access_key_id",
    aws_secret_access_key="aws_secret_access_key",
)

def load_model_from_s3(bucket_name, object_path):
    s3 = session.resource('s3')
    obj = s3.Bucket(bucket_name).Object(object_path)
    knn_model = pickle.loads(obj.get()['Body'].read())
    return knn_model

# Define Obejct Bucket path stored in S3
bucket_name= 'bucket_nam'
object_path= 'object_path'
# Load a trained KNN model 
knn_clf = load_model_from_s3(bucket_name, object_path)

def predict_faces(frame, resize_factor=0.5):
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

@socketio.on('image')
def image(data_image):

    # Define skip factor (adjust as needed)
    skip_factor = 4  # Process 1 frame every 5 frames

    # Frame counter (initialize to 0)
    frame_count = 0
   
    try:
        # Decode and convert into image
        b = io.BytesIO(base64.b64decode(data_image))
        pimg = Image.open(b)

        # Converting RGB to BGR, as OpenCV standards
        frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

        # Resize the frame (adjust resize_factor as needed)
        resize_factor = 1
        resized_frame = cv2.resize(frame, (int(frame.shape[1] * resize_factor), int(frame.shape[0] * resize_factor)))
        
        # Check if frame should be skipped based on skip_factor
        if frame_count % skip_factor != 0:
            frame_count += 1  # Increment counter even if skipped
            return  # Skip processing for this frame

        # Predict faces in the frame
        future = executor.submit(predict_faces, resized_frame)
        # Process the frame only if not skipped (original logic)
        frame_count += 1
        # Print predictions
        predictions = future.result()
        for name, (top, right, bottom, left) in predictions:
            print(f"Detected {name} at ({top}, {right}, {bottom}, {left})")

        # If you need to emit a response back, you can modify or remove this part as needed
        # For now, we just return without sending back the image
        return

    except Exception as e:
        print(f"Error processing image: {e}")

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=5000)

# usage : gunicorn --certfile=certificates/server.crt --keyfile=certificates/server.key -w 4 -b 0.0.0.0:5000 --worker-class eventlet facerec_rec_known:app
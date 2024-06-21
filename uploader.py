from flask import Flask, request, jsonify
from google.cloud import storage
from datetime import timedelta
import os
import uuid

app = Flask(__name__)

# Initialize the Google Cloud Storage client
storage_client = storage.Client()
image_bucket_name = os.environ.get('greenguard1')
bucket = storage_client.bucket(image_bucket_name)

@app.route('/upload-image', methods=['POST'])
def upload_image():
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'Image is required'}), 400,
    tmp_file = f'/tmp/{file.filename}'
    file.save(tmp_file)
    blob = gcs_upload_image(tmp_file)
    os.remove(tmp_file)
    return jsonify({'result': 'File uploaded successfully'})

def gcs_upload_image(filename: str):
    storage_client: storage.Client = storage.Client()
    bucket: storage.Bucket = storage_client.bucket('greenguard1')
    blob: storage.Blob = bucket.blob(filename.split("/")[-1])
    blob.upload_from_filename(filename)
    return blob

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

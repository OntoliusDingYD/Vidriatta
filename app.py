from flask import Flask, request, render_template, jsonify
from ultralytics import YOLO
import os

import boto3
import uuid

app = Flask(__name__)
model = YOLO('./models/yolov8n.pt')  # tiny模型
app.config['UPLOAD_FOLDER'] = 'static'

BUCKET_NAME = 'vidriatta-detection'  # replacable
s3_cli = boto3.client('s3')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filename = f"{uuid.uuid4()}.jpg"
    local_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(local_path)

    s3_key = f"uploads/{filename}"
    s3_cli.upload_file(local_path, BUCKET_NAME, s3_key)

    results = model(local_path)
    results[0].save(filename=os.path.join(app.config['UPLOAD_FOLDER'], f"result_{filename}"))
    
    detections = results[0].boxes
    result_list = []
    for box in detections:
        result_list.append({
            "class_id": int(box.cls[0]),
            "conf": round(float(box.conf[0]), 4),
            "bbox": [round(float(x), 2) for x in box.xyxy[0]]
        })

    return jsonify({
        'output_image_url': f"/static/result_{filename}",
        's3_key': s3_key,
        'detections': result_list
    })

if __name__ == "__main__":
    app.run(debug=True)
    #app.run(host='0.0.0.0', port=5000)  # 生产环境使用

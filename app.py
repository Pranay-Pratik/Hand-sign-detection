from flask import Flask, render_template, Response, jsonify, request
from utils.camera import VideoCamera

app = Flask(__name__)

camera = None
latest_prediction = "No sign detected yet"

def gen_frames(camera):
    global latest_prediction
    while True:
        frame = camera.get_frame()
        if frame is not None:
            latest_prediction = camera.get_prediction_text()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recognition')
def recognition():
    return render_template('recognition.html')

@app.route('/video_feed')
def video_feed():
    global camera
    if camera:
        return Response(gen_frames(camera),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    return '', 204

@app.route('/start', methods=['POST'])
def start_camera():
    global camera
    if not camera:
        camera = VideoCamera()
    camera.set_active(True)
    return '', 204

@app.route('/stop', methods=['POST'])
def stop_camera():
    global camera
    if camera:
        camera.set_active(False)
        camera.cap.release()
        camera = None
    return '', 204

@app.route('/get_prediction')
def get_prediction():
    global camera
    if camera:
        return jsonify({'text': camera.get_prediction_text()})
    return jsonify({'text': 'Camera not active'})

# 🆕 Get the full prediction history
@app.route('/get_history')
def get_history():
    global camera
    if camera:
        return jsonify({'history': camera.get_prediction_history()})
    return jsonify({'history': ''})

# 🆕 Reset the prediction history
@app.route('/reset_history', methods=['POST'])
def reset_history():
    global camera
    if camera:
        camera.reset_prediction_history()
        return jsonify({'status': 'History reset'})
    return jsonify({'status': 'Camera not active'}), 400

if __name__ == '__main__':
    app.run(debug=True)

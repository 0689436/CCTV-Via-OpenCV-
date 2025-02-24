from flask import Flask, Response, render_template
import cv2

app = Flask(__name__)

# Initialize background subtractor with adjusted parameters
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Function to capture video from USB webcam with motion detection
def usb_camera():
    cap = cv2.VideoCapture(0)  # 0 is usually the default USB webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply background subtraction
        fgmask = fgbg.apply(frame)
        _, fgmask = cv2.threshold(fgmask, 254, 255, cv2.THRESH_BINARY)

        # Find contours of moving objects
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw rectangles around moving objects
        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # Increased minimum contour area to reduce sensitivity
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle

        # Encode the frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Function to capture video from DroidCam with motion detection
def droidcam():
    # Replace with your DroidCam IP and port
    droidcam_url = "http://192.168.1.13:4747/video"
    cap = cv2.VideoCapture(droidcam_url)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply background subtraction
        fgmask = fgbg.apply(frame)
        _, fgmask = cv2.threshold(fgmask, 254, 255, cv2.THRESH_BINARY)

        # Find contours of moving objects
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw rectangles around moving objects
        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # Increased minimum contour area to reduce sensitivity
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle

        # Encode the frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed_usb')
def video_feed_usb():
    return Response(usb_camera(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_droidcam')
def video_feed_droidcam():
    return Response(droidcam(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

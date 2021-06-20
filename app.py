from flask import Flask,render_template,Response
import cv2

app=Flask(__name__)
camera=cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('indexvideo.html')

def generate_frames():
    while True:
        success,frame = camera.read()
        # read() returns two values
        # a boolean value to indicate if the video capture was successful or not
        # The frame

        if not success:
            break
        else:
            detector = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
            eye_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')
            faces = detector.detectMultiScale(frame,1.1,7)
            # detectMultiScale() returns x,y coordinates and width and height of the face
            # in a frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Draw the rectangle around each face
            for(x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_classifier.detectMultiScale(roi_gray,1.1,3)
                for(ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)



            ret, buffer = cv2.imencode('.jpg',frame)
            # We have to encode the frame into jpg or img or png format
            # imencode() returns
            frame = buffer.tobytes()
            # Have to convert the frame from compressed buffer memory into bytes

        yield(b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        # This yield() is an alterntive to return without breaking the loop

@app.route('/video')
def video():
    # Response is some kind of response we want to send in a continuous manner
    # This function will take all the frames from our webcam
    # and send them to "indexvideo.html" as response
    # since "indexvideo.html" is calling "video" route
    return Response(generate_frames(),mimetype="multipart/x-mixed-replace; boundary=frame")

    # Whenever we are passing a function we need to set some mimetype

if __name__ == "__main__":
    app.run(debug=True)

import pickle
import cv2
import os
import face_recognition
from flask import Flask,request,render_template,redirect,session,url_for
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import time
# import db

#VARIABLES
MESSAGE = "WELCOME  " \
          " Instruction: to register your attendence kindly click on 'a' on keyboard"

#### Defining Flask App
app = Flask(__name__)

#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
day = datetime.today().strftime("%A")
datetoday2 = date.today().strftime("%d-%B-%Y")
c_time = datetime.now().strftime("%H:%M")
known_images, known_encodings, known_names =[],[],[]
#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
try:
    cap = cv2.VideoCapture(1)
except:
    cap = cv2.VideoCapture(0)

#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv','w') as f:
        f.write('Name,Roll,Time')

#### get a number of total registered users

def totalreg():
    return len(os.listdir('static/faces'))

#### extract the face from an image
def extract_faces(img):
    if img!=[]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    else:
        return []

#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

#### A function which trains the model on all the faces available in faces folder
import os
import face_recognition

def load_images_from_directory(folder_path):
    images = []
    encodings = []
    names = []

    for root, _, files in os.walk(folder_path):
        # Process images from this subfolder
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Case-insensitive image extensions
                image_path = os.path.join(root, filename)
                try:
                    image = face_recognition.load_image_file(image_path)
                    encoding = face_recognition.face_encodings(image)[0]
                    name = os.path.splitext(filename)[0]  # Extract name from filename (without extension)
                    images.append(image)
                    encodings.append(encoding)
                    names.append(name)
                except FileNotFoundError:
                    print(f"Error: File not found: {image_path}")  # Handle missing files gracefully
                except IndexError:
                    print(f"Error: No faces detected in image: {image_path}")  # Handle empty images
    with open('encodings.pkl', 'wb') as f:
        pickle.dump((encodings, names), f)
    return images, encodings, names


# ... rest of your face recognition code using images, encodings, and names

#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names,rolls,times,l

#### Add Attendance of a specific user
def add_attendance(name):
  username = name.split('_')[0]
  userid = name.split('_')[1]
  current_time = datetime.now().strftime("%H:%M:%S")
  df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
  if username not in df['Name'].tolist():
    # New attendance entry, append to the DataFrame
    new_attendance = pd.DataFrame({'Name': [username], 'Roll': [userid], 'Time': [current_time]})
    df = pd.concat([df, new_attendance], ignore_index=True)
    # Save the updated attendance data
    df.to_csv(f'Attendance/Attendance-{datetoday}.csv', index=False)
    print(f"{username} ({userid}) marked attendance at {current_time}.")
  else:
    print(f"{username} ({userid}) has already marked attendance today.")
################## ROUTING FUNCTIONS ##############################

#### Our main page
@app.route('/')
def home():
    names,rolls,times,l = extract_attendance()
    return render_template('home.html',names=names,rolls=rolls,times=c_time,l=l,totalreg=totalreg(),datetoday2=datetoday2,day=day, mess = MESSAGE)


#### This function will run when we click on Take Attendance Button
@app.route('/start',methods=['GET'])
def start():

    with open('encodings.pkl', 'rb') as f:
        known_encodings, known_names = pickle.load(f)
    # Initialize video capture and face recognition process
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()

        # Convert frame to RGB format for face recognition
        rgb_frame = frame[:, :, ::-1]

        # Find all faces and encodings in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through each detected face
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Match the face encoding with the known face encodings
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"

            # If a match is found, get the name
            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]

                # Mark attendance for the recognized person
                add_attendance(name)

            # Draw rectangle around the face and display name
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Display name below the face rectangle
            font_scale = 1.0
            font_thickness = 2
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255, 255, 255), font_thickness)

        # Display the resulting frame
        cv2.imshow('Attendance System', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close windows
    video_capture.release()
    cv2.destroyAllWindows()

    names, rolls, times, l = extract_attendance()
    MESSAGE = 'Attendence taken successfully'
    print("attendence registered")
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2, mess=MESSAGE)

@app.route('/add',methods=['GET','POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i,j = 0,0
    while 1:
        _,frame = cap.read()
        faces = extract_faces(frame)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame,f'Images Captured: {i}/50',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
            if j%10==0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name,frame[y:y+h,x:x+w])
                i+=1
            j+=1
        if j==500:
            break
        cv2.imshow('Adding new User',frame)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    known_images_folder = 'static/faces'  # Replace with your actual folder path
    load_images_from_directory(known_images_folder)
    names,rolls,times,l = extract_attendance()
    if totalreg() > 0 :
        names, rolls, times, l = extract_attendance()
        MESSAGE = 'User added Sucessfully'
        print("message changed")
        return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2, mess = MESSAGE)
    else:
        return redirect(url_for('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2))
    # return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2)

#### Our main function which runs the Flask App
app.run(debug=True,port=1000)
if __name__ == '__main__':
    pass
#### This function will run when we add a new user

import cv2
import os
from flask import Flask, request, render_template, redirect, url_for, flash
from datetime import date
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import glob
import face_recognition
import os
import pickle



# Defining Flask App
app = Flask(__name__)
app.secret_key = '12345678'


nimgs = 10

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-Period1-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-Period1-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')


# get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


# extract the face from an image
def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 6, minSize=(20, 20))
        return face_points
    except:
        return []


# Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


# A function which trains the model on all the faces available in faces folder
def train_model():
    # The directory containing photos of people
    dir_path = "static/faces"

    # A dictionary where the keys are people's names and the values are lists of face encodings
    face_encodings_dict = {}

    for user in os.listdir(dir_path):
        for imgname in os.listdir(f'static/faces/{user}'):
            # Extract the person's name from the image name
            person_name = user

            # Load the image
            image_path = os.path.join(dir_path, user, imgname)
            image = face_recognition.load_image_file(image_path)

            # Find all face encodings in the image
            encodings = face_recognition.face_encodings(image)

            # Add the encodings to our dictionary
            if person_name in face_encodings_dict:
                face_encodings_dict[person_name].extend(encodings)
            else:
                face_encodings_dict[person_name] = encodings

    # Save the face encodings dictionary to a file
    with open("static/face_encodings.pkl", "wb") as f:
        pickle.dump(face_encodings_dict, f)

# Extract info from today's attendance file in attendance folder
def extract_attendance():
    # Find the latest attendance file
    latest_attendance_file = max(glob.glob('Attendance/Attendance-*.csv'), key=os.path.getctime)

    df = pd.read_csv(latest_attendance_file)
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l

def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    # Find the latest attendance file
    latest_attendance_file = max(glob.glob('Attendance/Attendance-*.csv'), key=os.path.getctime)

    df = pd.read_csv(latest_attendance_file)
    if int(userid) not in list(df['Roll']):
        with open(latest_attendance_file, 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')

## A function to get names and rol numbers of all users
def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l


## A function to delete a user folder 
def deletefolder(duser):
    pics = os.listdir(duser)
    for i in pics:
        os.remove(duser+'/'+i)
    os.rmdir(duser)




################## ROUTING FUNCTIONS #########################

# Our main page
@app.route('/')
def home():
    # Find the latest attendance file
    latest_attendance_file = max(glob.glob('Attendance/Attendance-*.csv'), key=os.path.getctime)

    # Extract attendance information from the latest file
    df = pd.read_csv(latest_attendance_file)
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)

    return render_template('index.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

## List users page
@app.route('/listusers')
def listusers():
    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)


## Delete functionality
@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    duser = request.args.get('user')
    deletefolder('static/faces/'+duser)

    ## if all the face are deleted, delete the trained file...
    if os.listdir('static/faces/')==[]:
        os.remove('static/face_recognition_model.pkl')
    
    try:
        train_model()
    except:
        pass

    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)


# Our main Face Recognition functionality. 
# This function will run when we click on Take Attendance Button.
@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()

    if 'face_encodings.pkl' not in os.listdir('static/'):
        return render_template('index.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

    # Load the face encodings dictionary from the file
    with open("static/face_encodings.pkl", "rb") as f:
        face_encodings_dict = pickle.load(f)

    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            for person_name, person_face_encodings in face_encodings_dict.items():
                # Calculate the face distance
                face_distances = face_recognition.face_distance(person_face_encodings, face_encoding)

                # Find the best match
                best_match_index = np.argmin(face_distances)

                # If the best match is below the threshold, display the detected person
                if face_distances[best_match_index] < 0.3:  # Adjust this value based on your requirement
                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                    # Draw a label with a name below the face
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, person_name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                    # Add attendance
                    add_attendance(person_name)

        # Display the resulting image
        cv2.imshow('Attendance', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('index.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)
# A function to add a new user.
# This function will run when we add a new user.


@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']

    # Check if a user with the same ID already exists
    existing_users = os.listdir('static/faces')
    for user in existing_users:
        _, existing_userid = user.split('_')
        if existing_userid == newuserid:
            flash('User with the same ID already exists. Please choose a different ID or delete the existing user.')
            return redirect(url_for('home'))

    userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)

    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = newusername + '_' + str(i) + '.jpg'
                cv2.imwrite(userimagefolder + '/' + name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == nimgs*5:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, times, l = extract_attendance()
    return render_template('index.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

# Route to create a new period (new attendance file) and clear the attendance list
@app.route('/newperiod', methods=['GET'])
def newperiod():
    # Increment the period number
    period_number = len([file for file in os.listdir('Attendance') if file.startswith('Attendance-')])

    # Create a new attendance file for the new period
    new_attendance_file = f'Attendance/Attendance-Period{period_number + 1}-{datetoday}.csv'
    with open(new_attendance_file, 'w') as f:
        f.write('Name,Roll,Time')

    # Redirect back to the home page
    return redirect(url_for('home'))

# Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)

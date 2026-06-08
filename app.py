import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

import cv2
from deepface import DeepFace
import numpy as np
import openpyxl
from sklearn.neighbors import NearestNeighbors
from datetime import datetime
from pymongo import MongoClient
from bson.objectid import ObjectId
from flask import Flask, render_template, request, redirect, url_for, session, flash, Response, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import io

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))
app.config['UPLOAD_FOLDER'] = 'static/uploads/persons'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ================= MongoDB =================
client = MongoClient("mongodb://localhost:27017/")
db = client["attendance"]
students_collection = db["students"]
staff_collection = db["staff"]
users_collection = db["users"]
records_collection = db["records"]

# Initialize admin user if not exists
if not users_collection.find_one({"username": "admin"}):
    users_collection.insert_one({
        "username": "admin",
        "password": generate_password_hash("admin123")
    })

# Load students function
def load_known_faces():
    known_faces = []
    embeddings_list = []
    for student in students_collection.find():
        if student.get("embedding"):
            known_faces.append({
                "id": str(student["_id"]),
                "name": student.get("name"),
                "student_id": student.get("student_id"),
                "role": student.get("role", "Student"),
                "level": student.get("level")
            })
            embeddings_list.append(student.get("embedding"))
            
    for staff in staff_collection.find():
        if staff.get("embedding"):
            known_faces.append({
                "id": str(staff["_id"]),
                "name": staff.get("name"),
                "student_id": staff.get("student_id"),
                "role": staff.get("role", "Staff"),
                "level": staff.get("level")
            })
            embeddings_list.append(staff.get("embedding"))
            
    if embeddings_list:
        embeddings_matrix = np.array(embeddings_list)
        nn_model = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric='euclidean')
        nn_model.fit(embeddings_matrix)
    else:
        nn_model = None
        
    return known_faces, nn_model

# ================= Global State =================
known_faces, nn_model = load_known_faces()
camera_active = False

# ================= Attendance Functions =================
def mark_attendance(name, student_id, level, role="Student"):
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Check database for today's attendance
    existing = records_collection.find_one({
        "student_id": student_id,
        "date": today
    })

    if not existing:
        now = datetime.now()
        time = now.strftime("%I:%M %p")
        
        # Save to DB for records page
        records_collection.insert_one({
            "person_name": name,
            "student_id": student_id,
            "role": role,
            "level": level,
            "date": today,
            "time": time,
            "status": "Present",
            "created_at": now
        })
        
        return True

    return False

# ================= Camera Generator =================
def generate_frames():
    global camera_active
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    cap = cv2.VideoCapture(0)

    timer = 0
    message = ""
    frame_count = 0
    process_every_n_frames = 5
    last_faces_data = []

    while camera_active:
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        
        if frame_count % process_every_n_frames == 0:
            last_faces_data = [] # Reset for this frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]

                name = "Unknown"
                student_id = ""
                role = "Student"
                level = ""

                try:
                    emb = DeepFace.represent(face_img, enforce_detection=False)[0]["embedding"]

                    if nn_model is not None:
                        distances, indices = nn_model.kneighbors([emb])
                        
                        if distances[0][0] < 1:
                            match = known_faces[indices[0][0]]
                            name = match["name"]
                            student_id = match["student_id"]
                            role = match.get("role", "Student")
                            level = match["level"]

                    if name != "Unknown":
                        if mark_attendance(name, student_id, level, role):
                            message = f"{name} Recorded"
                            timer = 50

                except Exception as e:
                    print(f"Face Recognition Error: {e}")

                last_faces_data.append((x, y, w, h, name))

        for (x, y, w, h, name) in last_faces_data:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if timer > 0:
            cv2.putText(frame, message, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            timer -= 1

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
    cap.release()

# ================= Routes =================
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = users_collection.find_one({"username": username})
        if user:
            stored_password = user.get('password') or user.get('password_hash')
            if stored_password and check_password_hash(stored_password, password):
                session['user_id'] = str(user['_id'])
                session['username'] = user['username']
                flash('Logged in successfully', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid username or password', 'error')
        else:
            flash('Invalid username or password', 'error')
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session: return redirect(url_for('login'))
    
    total_persons = students_collection.count_documents({}) + staff_collection.count_documents({})
    today = datetime.now().strftime("%Y-%m-%d")
    today_attendance = records_collection.count_documents({"date": today})
    total_attendance = records_collection.count_documents({})
    
    stats = {
        "total_persons": total_persons,
        "today_attendance": today_attendance,
        "total_attendance": total_attendance
    }
    return render_template('dashboard.html', stats=stats)

@app.route('/attendance_camera')
def attendance_camera():
    if 'user_id' not in session: return redirect(url_for('login'))
    return render_template('attendance.html', camera_active=camera_active)

@app.route('/start_camera')
def start_camera():
    if 'user_id' not in session: return redirect(url_for('login'))
    global camera_active
    camera_active = True
    return redirect(url_for('attendance_camera'))

@app.route('/stop_camera')
def stop_camera():
    if 'user_id' not in session: return redirect(url_for('login'))
    global camera_active
    camera_active = False
    return redirect(url_for('attendance_camera'))

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/add_person', methods=['GET', 'POST'])
def add_person():
    if 'user_id' not in session: return redirect(url_for('login'))
    global known_faces, nn_model
    
    if request.method == 'POST':
        raw_id = request.form['student_id'].strip()
        name = request.form['full_name']
        level = request.form['level']
        role = request.form.get('role', 'Student')
        
        if not raw_id.isdigit():
            flash('ID must be a number.', 'error')
            return redirect(request.url)
            
        formatted_num = raw_id.zfill(4)
        
        if role == 'Student':
            student_id = f"STU-{formatted_num}"
        elif role == 'Doctor':
            student_id = f"DOC-{formatted_num}"
        elif role == 'Teaching Assistant':
            student_id = f"TA-{formatted_num}"
        else:
            student_id = f"STAFF-{formatted_num}"
            
        if students_collection.find_one({"student_id": student_id}) or staff_collection.find_one({"student_id": student_id}):
            flash('This ID already exists. Please enter another number.', 'error')
            return redirect(request.url)
        
        captured_image_data = request.form.get('captured_image')
        file = request.files.get('image')
        
        filepath = None
        filename = None

        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
        elif captured_image_data:
            import base64
            if "base64," in captured_image_data:
                captured_image_data = captured_image_data.split("base64,")[1]
            
            filename = f"capture_{student_id}_{int(datetime.now().timestamp())}.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(filepath, "wb") as f:
                f.write(base64.b64decode(captured_image_data))
        else:
            flash('No image provided. Please upload a file or capture a photo.', 'error')
            return redirect(request.url)
            
        try:
            emb = DeepFace.represent(filepath, enforce_detection=False)[0]["embedding"]
                
            collection = students_collection if role == 'Student' else staff_collection
            collection.insert_one({
                "student_id": student_id,
                "name": name,
                "role": role,
                "level": level,
                "image_path": filename,
                "embedding": emb,
                "created_at": datetime.now()
            })
            
            known_faces, nn_model = load_known_faces()
            flash('Person added successfully', 'success')
            return redirect(url_for('list_persons'))
        except Exception as e:
            flash(f'Error processing face: {str(e)}', 'error')
                
    return render_template('add_person.html')

@app.route('/persons')
def list_persons():
    if 'user_id' not in session: return redirect(url_for('login'))
    
    persons = []
    for p in students_collection.find():
        persons.append({
            "id": str(p["_id"]),
            "student_id": p.get("student_id"),
            "full_name": p.get("name"),
            "role": p.get("role", "Student"),
            "level": p.get("level"),
            "image_path": p.get("image_path"),
            "created_at": p.get("created_at", datetime.now())
        })
    for p in staff_collection.find():
        persons.append({
            "id": str(p["_id"]),
            "student_id": p.get("student_id"),
            "full_name": p.get("name"),
            "role": p.get("role", "Staff"),
            "level": p.get("level"),
            "image_path": p.get("image_path"),
            "created_at": p.get("created_at", datetime.now())
        })
    role_order = {
        'Doctor': 1,
        'Teaching Assistant': 2,
        'Student': 3
    }
    persons.sort(key=lambda x: (role_order.get(x.get('role'), 4), x.get('created_at', datetime.min)))
    return render_template('persons.html', persons=persons)

@app.route('/delete_person/<person_id>', methods=['POST'])
def delete_person(person_id):
    if 'user_id' not in session: return redirect(url_for('login'))
    global known_faces, nn_model
    
    person = students_collection.find_one({"_id": ObjectId(person_id)})
    collection = students_collection
    if not person:
        person = staff_collection.find_one({"_id": ObjectId(person_id)})
        collection = staff_collection
        
    if person:
        if person.get('image_path'):
            path = os.path.join(app.config['UPLOAD_FOLDER'], person['image_path'])
            if os.path.exists(path):
                os.remove(path)
        collection.delete_one({"_id": ObjectId(person_id)})
        known_faces, nn_model = load_known_faces()
        flash('Person deleted successfully', 'success')
        
    return redirect(url_for('list_persons'))

@app.route('/records')
def list_records():
    if 'user_id' not in session: return redirect(url_for('login'))
    
    search_name = request.args.get('name', '')
    filter_date = request.args.get('date', '')
    
    query = {}
    if search_name:
        query['person_name'] = {'$regex': search_name, '$options': 'i'}
    if filter_date:
        query['date'] = filter_date
        
    records = list(records_collection.find(query).sort("created_at", -1))
    for r in records:
        r['id'] = str(r['_id'])
        if 'created_at' not in r:
            r['created_at'] = datetime.now()
            
    return render_template('records.html', records=records, search_name=search_name, filter_date=filter_date)

@app.route('/delete_record/<record_id>', methods=['POST'])
def delete_record(record_id):
    if 'user_id' not in session: return redirect(url_for('login'))
    
    records_collection.delete_one({"_id": ObjectId(record_id)})
    flash('Attendance record deleted successfully from the database.', 'success')
    return redirect(url_for('list_records'))

@app.route('/export_excel')
def export_excel():
    if 'user_id' not in session: return redirect(url_for('login'))
    
    today = datetime.now().strftime("%Y-%m-%d")
    records = list(records_collection.find({"date": today}))
    
    wb = openpyxl.Workbook()
    sheet = wb.active
    sheet.append(["Name", "ID", "Role", "Level/Dept", "Date", "Time", "Status"])
    
    for r in records:
        sheet.append([
            r.get("person_name", ""),
            r.get("student_id", ""),
            r.get("role", ""),
            r.get("level", ""),
            r.get("date", ""),
            r.get("time", ""),
            r.get("status", "")
        ])
        
    excel_file = io.BytesIO()
    wb.save(excel_file)
    excel_file.seek(0)
    
    filename = f"attendance_{today}.xlsx"
    return send_file(excel_file, as_attachment=True, download_name=filename, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')

import os
import io
import re
import base64
import binascii
import secrets
from datetime import datetime
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

import cv2
from deepface import DeepFace
import numpy as np
import openpyxl

from sklearn.neighbors import NearestNeighbors
from pymongo import MongoClient
from bson.objectid import ObjectId
from pymongo.errors import PyMongoError

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    Response,
    send_file,
)

from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename


# ================= Flask Configuration =================

app = Flask(__name__)

APP_ENV = os.environ.get("APP_ENV", "development").lower()
IS_PRODUCTION = APP_ENV == "production"

secret_key = os.environ.get("SECRET_KEY")

if IS_PRODUCTION and not secret_key:
    raise RuntimeError(
        "SECRET_KEY must be configured when APP_ENV=production."
    )

app.secret_key = secret_key or secrets.token_hex(32)

app.config["UPLOAD_FOLDER"] = os.path.join(
    "static", "uploads", "persons"
)

# Limit uploads to 5 MB
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=IS_PRODUCTION,
)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png"}


# ================= MongoDB =================

MONGO_URI = os.environ.get(
    "MONGO_URI",
    "mongodb://localhost:27017/"
)

client = MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=5000,
)

db = client["attendance"]

students_collection = db["students"]
staff_collection = db["staff"]
users_collection = db["users"]
records_collection = db["records"]


# ================= Admin Initialization =================
# No default username/password is created.
# Set ADMIN_USERNAME and ADMIN_PASSWORD before starting the app
# to create the configured admin account if it does not exist.

admin_username = os.environ.get("ADMIN_USERNAME")
admin_password = os.environ.get("ADMIN_PASSWORD")

if admin_username and admin_password:
    existing_admin = users_collection.find_one(
        {"username": admin_username}
    )

    if not existing_admin:
        users_collection.insert_one({
            "username": admin_username,
            "password": generate_password_hash(admin_password),
        })

elif users_collection.count_documents({}) == 0:
    app.logger.warning(
        "No admin account configured. Set ADMIN_USERNAME "
        "and ADMIN_PASSWORD before logging in."
    )


# ================= Authentication Helper =================

def is_logged_in():
    return bool(session.get("user_id"))


def login_required_redirect():
    return redirect(url_for("login"))


# ================= Image Helpers =================

def allowed_image(filename):
    if not filename or "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1].lower()
    return extension in ALLOWED_IMAGE_EXTENSIONS


def save_uploaded_image(file):
    """
    Save an uploaded image using a unique filename.
    Returns the saved filename, or None if invalid.
    """
    if not file or not file.filename:
        return None

    safe_name = secure_filename(file.filename)

    if not safe_name or not allowed_image(safe_name):
        raise ValueError(
            "Please upload a JPG, JPEG, or PNG image."
        )

    extension = safe_name.rsplit(".", 1)[1].lower()
    filename = f"{secrets.token_hex(16)}.{extension}"
    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename,
    )

    file_bytes = file.read()

    if not file_bytes:
        raise ValueError("The uploaded image is empty.")

    if len(file_bytes) > app.config["MAX_CONTENT_LENGTH"]:
        raise ValueError("The image exceeds the 5 MB size limit.")

    # Validate that the file can actually be decoded as an image.
    image_array = np.frombuffer(file_bytes, dtype=np.uint8)
    decoded_image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if decoded_image is None:
        raise ValueError("The uploaded file is not a valid image.")

    if decoded_image.size == 0:
        raise ValueError("The uploaded image is invalid.")

    with open(filepath, "wb") as output_file:
        output_file.write(file_bytes)

    return filename


def save_captured_image(captured_image_data, person_id):
    """
    Decode a base64 data URL and save a validated image.
    """
    if not captured_image_data:
        return None

    try:
        if "," in captured_image_data:
            header, encoded_data = captured_image_data.split(",", 1)

            if not header.startswith("data:image/"):
                raise ValueError("Invalid captured image format.")

            if ";base64" not in header:
                raise ValueError("Invalid captured image encoding.")
        else:
            encoded_data = captured_image_data

        image_bytes = base64.b64decode(
            encoded_data,
            validate=True,
        )

        if not image_bytes:
            raise ValueError("The captured image is empty.")

        if len(image_bytes) > app.config["MAX_CONTENT_LENGTH"]:
            raise ValueError("The captured image exceeds 5 MB.")

        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        decoded_image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR,
        )

        if decoded_image is None or decoded_image.size == 0:
            raise ValueError("The captured image is invalid.")

        filename = (
            f"capture_{secrets.token_hex(12)}.jpg"
        )

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename,
        )

        if not cv2.imwrite(filepath, decoded_image):
            raise ValueError("Could not save the captured image.")

        return filename

    except (binascii.Error, ValueError):
        raise ValueError("Could not decode the captured image.")


def remove_person_image(filename):
    """
    Remove an image only if its resolved path is inside
    the configured uploads directory.
    """
    if not filename:
        return

    upload_dir = Path(
        app.config["UPLOAD_FOLDER"]
    ).resolve()

    image_path = (upload_dir / filename).resolve()

    if image_path.parent != upload_dir:
        app.logger.warning("Rejected an unsafe image path.")
        return

    if image_path.is_file():
        image_path.unlink()


# ================= Load Known Faces =================

def load_known_faces():
    known_faces = []
    embeddings_list = []

    collections = [
        (students_collection, "Student"),
        (staff_collection, "Staff"),
    ]

    for collection, default_role in collections:
        for person in collection.find():
            embedding = person.get("embedding")

            if not embedding:
                continue

            try:
                embedding_array = np.asarray(
                    embedding,
                    dtype=np.float32,
                )

                if (
                    embedding_array.ndim != 1
                    or embedding_array.size == 0
                    or not np.all(np.isfinite(embedding_array))
                ):
                    app.logger.warning(
                        "Skipping an invalid face embedding."
                    )
                    continue

                known_faces.append({
                    "id": str(person["_id"]),
                    "name": person.get("name", "Unknown"),
                    "student_id": person.get("student_id", ""),
                    "role": person.get("role", default_role),
                    "level": person.get("level", ""),
                })

                embeddings_list.append(embedding_array)

            except (TypeError, ValueError):
                app.logger.warning(
                    "Skipping an unreadable face embedding."
                )

    if embeddings_list:
        embeddings_matrix = np.asarray(
            embeddings_list,
            dtype=np.float32,
        )

        nn_model = NearestNeighbors(
            n_neighbors=1,
            algorithm="auto",
            metric="euclidean",
        )

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

    existing = records_collection.find_one({
        "student_id": student_id,
        "date": today,
    })

    if existing:
        return False

    now = datetime.now()

    time_string = now.strftime("%I:%M %p")

    records_collection.insert_one({
        "person_name": name,
        "student_id": student_id,
        "role": role,
        "level": level,
        "date": today,
        "time": time_string,
        "status": "Present",
        "created_at": now,
    })

    return True


# ================= Camera Generator =================

def generate_frames():
    global camera_active

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    if face_cascade.empty():
        app.logger.error("Could not load the face cascade.")
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        app.logger.error("Could not open the camera.")
        cap.release()
        return

    timer = 0
    message = ""
    frame_count = 0
    process_every_n_frames = 5
    last_faces_data = []

    try:
        while camera_active:
            success, frame = cap.read()

            if not success:
                break

            frame_count += 1

            if frame_count % process_every_n_frames == 0:
                last_faces_data = []

                gray = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2GRAY,
                )

                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.3,
                    minNeighbors=5,
                )

                for (x, y, w, h) in faces:
                    face_img = frame[y:y + h, x:x + w]

                    name = "Unknown"
                    student_id = ""
                    role = "Student"
                    level = ""

                    try:
                        embedding_result = DeepFace.represent(
                            face_img,
                            enforce_detection=False,
                        )

                        if not embedding_result:
                            last_faces_data.append(
                                (x, y, w, h, name)
                            )
                            continue

                        embedding = np.asarray(
                            embedding_result[0]["embedding"],
                            dtype=np.float32,
                        )

                        if (
                            nn_model is not None
                            and embedding.ndim == 1
                            and np.all(np.isfinite(embedding))
                        ):
                            distances, indices = (
                                nn_model.kneighbors([embedding])
                            )

                            if distances[0][0] < 1:
                                match_index = indices[0][0]

                                if match_index < len(known_faces):
                                    match = known_faces[match_index]

                                    name = match["name"]
                                    student_id = match["student_id"]
                                    role = match.get(
                                        "role",
                                        "Student",
                                    )
                                    level = match.get("level", "")

                        if name != "Unknown" and student_id:
                            if mark_attendance(
                                name,
                                student_id,
                                level,
                                role,
                            ):
                                message = f"{name} Recorded"
                                timer = 50

                    except Exception:
                        # Keep the camera running if recognition fails.
                        app.logger.exception(
                            "Face recognition or attendance failed."
                        )

                    last_faces_data.append(
                        (x, y, w, h, name)
                    )

            for (x, y, w, h, name) in last_faces_data:
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    name,
                    (x, max(y - 5, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            if timer > 0:
                cv2.putText(
                    frame,
                    message,
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3,
                )

                timer -= 1

            ret, buffer = cv2.imencode(".jpg", frame)

            if not ret:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )

    finally:
        cap.release()


# ================= Routes =================

@app.route("/")
def index():
    if is_logged_in():
        return redirect(url_for("dashboard"))

    return redirect(url_for("login"))


# ================= Login =================

@app.route("/login", methods=["GET", "POST"])
def login():
    if is_logged_in():
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or not password:
            flash("Please enter your username and password.", "error")
            return render_template("login.html")

        try:
            user = users_collection.find_one({
                "username": username
            })

            if user:
                stored_password = (
                    user.get("password")
                    or user.get("password_hash")
                )

                if (
                    stored_password
                    and check_password_hash(
                        stored_password,
                        password,
                    )
                ):
                    # Clear old session values before creating
                    # the authenticated session.
                    session.clear()
                    session["user_id"] = str(user["_id"])
                    session["username"] = user["username"]

                    flash("Logged in successfully", "success")
                    return redirect(url_for("dashboard"))

            flash("Invalid username or password", "error")

        except PyMongoError:
            app.logger.exception("Login database error.")
            flash("Login is temporarily unavailable.", "error")

    return render_template("login.html")


@app.route("/logout", methods=["GET", "POST"])
def logout():
    session.clear()
    flash("Logged out successfully", "success")
    return redirect(url_for("login"))


# ================= Dashboard =================

@app.route("/dashboard")
def dashboard():
    if not is_logged_in():
        return login_required_redirect()

    try:
        total_persons = (
            students_collection.count_documents({})
            + staff_collection.count_documents({})
        )

        today = datetime.now().strftime("%Y-%m-%d")

        today_attendance = records_collection.count_documents({
            "date": today
        })

        total_attendance = records_collection.count_documents({})

        stats = {
            "total_persons": total_persons,
            "today_attendance": today_attendance,
            "total_attendance": total_attendance,
        }

        return render_template(
            "dashboard.html",
            stats=stats,
        )

    except PyMongoError:
        app.logger.exception("Dashboard database error.")
        flash("Could not load dashboard data.", "error")
        return render_template(
            "dashboard.html",
            stats={
                "total_persons": 0,
                "today_attendance": 0,
                "total_attendance": 0,
            },
        )


# ================= Attendance Camera =================

@app.route("/attendance_camera")
def attendance_camera():
    if not is_logged_in():
        return login_required_redirect()

    return render_template(
        "attendance.html",
        camera_active=camera_active,
    )


@app.route("/start_camera", methods=["GET", "POST"])
def start_camera():
    if not is_logged_in():
        return login_required_redirect()

    global camera_active
    camera_active = True

    return redirect(url_for("attendance_camera"))


@app.route("/stop_camera", methods=["GET", "POST"])
def stop_camera():
    if not is_logged_in():
        return login_required_redirect()

    global camera_active
    camera_active = False

    return redirect(url_for("attendance_camera"))


@app.route("/video_feed")
def video_feed():
    # Protect the video stream from unauthenticated access.
    if not is_logged_in():
        return login_required_redirect()

    if not camera_active:
        return (
            "Camera is not active. Start the camera first.",
            400,
        )

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


# ================= Add Person =================

@app.route("/add_person", methods=["GET", "POST"])
def add_person():
    if not is_logged_in():
        return login_required_redirect()

    global known_faces, nn_model

    if request.method == "POST":
        raw_id = request.form.get("student_id", "").strip()
        name = request.form.get("full_name", "").strip()
        level = request.form.get("level", "").strip()
        role = request.form.get("role", "Student").strip()

        if not raw_id.isdigit():
            flash("ID must be a number.", "error")
            return redirect(url_for("add_person"))

        if not name or len(name) > 150:
            flash("Please enter a valid name.", "error")
            return redirect(url_for("add_person"))

        if len(level) > 100:
            flash("Please enter a valid level or department.", "error")
            return redirect(url_for("add_person"))

        allowed_roles = {
            "Student",
            "Doctor",
            "Teaching Assistant",
            "Staff",
        }

        if role not in allowed_roles:
            flash("Invalid role selected.", "error")
            return redirect(url_for("add_person"))

        formatted_num = raw_id.zfill(4)

        if role == "Student":
            person_id = f"STU-{formatted_num}"
        elif role == "Doctor":
            person_id = f"DOC-{formatted_num}"
        elif role == "Teaching Assistant":
            person_id = f"TA-{formatted_num}"
        else:
            person_id = f"STAFF-{formatted_num}"

        try:
            if (
                students_collection.find_one({
                    "student_id": person_id
                })
                or staff_collection.find_one({
                    "student_id": person_id
                })
            ):
                flash(
                    "This ID already exists. Please enter another number.",
                    "error",
                )
                return redirect(url_for("add_person"))

            captured_image_data = request.form.get(
                "captured_image",
                "",
            )

            uploaded_file = request.files.get("image")
            filename = None

            if uploaded_file and uploaded_file.filename:
                filename = save_uploaded_image(uploaded_file)

            elif captured_image_data:
                filename = save_captured_image(
                    captured_image_data,
                    person_id,
                )

            else:
                flash(
                    "No image provided. Upload a file or capture a photo.",
                    "error",
                )
                return redirect(url_for("add_person"))

            if not filename:
                flash("Could not save the image.", "error")
                return redirect(url_for("add_person"))

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename,
            )

            try:
                embedding_result = DeepFace.represent(
                    filepath,
                    enforce_detection=True,
                )

                if not embedding_result:
                    raise ValueError(
                        "No face embedding could be generated."
                    )

                embedding = embedding_result[0]["embedding"]

                collection = (
                    students_collection
                    if role == "Student"
                    else staff_collection
                )

                collection.insert_one({
                    "student_id": person_id,
                    "name": name,
                    "role": role,
                    "level": level,
                    "image_path": filename,
                    "embedding": embedding,
                    "created_at": datetime.now(),
                })

            except Exception:
                # Remove the image if processing or insertion fails.
                remove_person_image(filename)
                raise

            known_faces, nn_model = load_known_faces()

            flash("Person added successfully", "success")
            return redirect(url_for("list_persons"))

        except ValueError as error:
            flash(str(error), "error")

        except PyMongoError:
            app.logger.exception("Database error while adding a person.")
            flash("Could not save the person to the database.", "error")

        except Exception:
            app.logger.exception("Error processing the person's face.")
            flash(
                "Could not process the image. Please try another clear photo.",
                "error",
            )

    return render_template("add_person.html")


# ================= List Persons =================

@app.route("/persons")
def list_persons():
    if not is_logged_in():
        return login_required_redirect()

    try:
        persons = []

        for person in students_collection.find():
            persons.append({
                "id": str(person["_id"]),
                "student_id": person.get("student_id"),
                "full_name": person.get("name"),
                "role": person.get("role", "Student"),
                "level": person.get("level"),
                "image_path": person.get("image_path"),
                "created_at": person.get(
                    "created_at",
                    datetime.min,
                ),
            })

        for person in staff_collection.find():
            persons.append({
                "id": str(person["_id"]),
                "student_id": person.get("student_id"),
                "full_name": person.get("name"),
                "role": person.get("role", "Staff"),
                "level": person.get("level"),
                "image_path": person.get("image_path"),
                "created_at": person.get(
                    "created_at",
                    datetime.min,
                ),
            })

        role_order = {
            "Doctor": 1,
            "Teaching Assistant": 2,
            "Student": 3,
        }

        persons.sort(
            key=lambda person: (
                role_order.get(person.get("role"), 4),
                person.get("created_at", datetime.min),
            )
        )

        return render_template(
            "persons.html",
            persons=persons,
        )

    except PyMongoError:
        app.logger.exception("Error loading persons.")
        flash("Could not load the persons list.", "error")
        return render_template("persons.html", persons=[])


# ================= Delete Person =================

@app.route("/delete_person/<person_id>", methods=["POST"])
def delete_person(person_id):
    if not is_logged_in():
        return login_required_redirect()

    global known_faces, nn_model

    if not ObjectId.is_valid(person_id):
        flash("Invalid person ID.", "error")
        return redirect(url_for("list_persons"))

    object_id = ObjectId(person_id)

    try:
        person = students_collection.find_one({
            "_id": object_id
        })

        collection = students_collection

        if not person:
            person = staff_collection.find_one({
                "_id": object_id
            })
            collection = staff_collection

        if person:
            collection.delete_one({"_id": object_id})

            if person.get("image_path"):
                try:
                    remove_person_image(person["image_path"])
                except OSError:
                    app.logger.exception(
                        "Could not remove the person's image."
                    )

            known_faces, nn_model = load_known_faces()
            flash("Person deleted successfully", "success")

        else:
            flash("Person not found.", "error")

    except PyMongoError:
        app.logger.exception("Error deleting a person.")
        flash("Could not delete the person.", "error")

    return redirect(url_for("list_persons"))


# ================= Attendance Records =================

@app.route("/records")
def list_records():
    if not is_logged_in():
        return login_required_redirect()

    search_name = request.args.get("name", "").strip()
    filter_date = request.args.get("date", "").strip()

    query = {}

    if search_name:
        # Escape regex characters to prevent unintended regex patterns.
        query["person_name"] = {
            "$regex": re.escape(search_name),
            "$options": "i",
        }

    if filter_date:
        try:
            datetime.strptime(filter_date, "%Y-%m-%d")
            query["date"] = filter_date
        except ValueError:
            flash("Invalid date filter.", "error")
            filter_date = ""

    try:
        records = list(
            records_collection.find(query).sort(
                "created_at",
                -1,
            )
        )

        for record in records:
            record["id"] = str(record["_id"])

            if "created_at" not in record:
                record["created_at"] = datetime.now()

        return render_template(
            "records.html",
            records=records,
            search_name=search_name,
            filter_date=filter_date,
        )

    except PyMongoError:
        app.logger.exception("Error loading attendance records.")
        flash("Could not load attendance records.", "error")

        return render_template(
            "records.html",
            records=[],
            search_name=search_name,
            filter_date=filter_date,
        )


# ================= Delete Attendance Record =================

@app.route("/delete_record/<record_id>", methods=["POST"])
def delete_record(record_id):
    if not is_logged_in():
        return login_required_redirect()

    if not ObjectId.is_valid(record_id):
        flash("Invalid attendance record ID.", "error")
        return redirect(url_for("list_records"))

    try:
        result = records_collection.delete_one({
            "_id": ObjectId(record_id)
        })

        if result.deleted_count:
            flash(
                "Attendance record deleted successfully from the database.",
                "success",
            )
        else:
            flash("Attendance record not found.", "error")

    except PyMongoError:
        app.logger.exception("Error deleting an attendance record.")
        flash("Could not delete the attendance record.", "error")

    return redirect(url_for("list_records"))


# ================= Export Attendance to Excel =================

@app.route("/export_excel")
def export_excel():
    if not is_logged_in():
        return login_required_redirect()

    today = datetime.now().strftime("%Y-%m-%d")

    try:
        records = list(
            records_collection.find({
                "date": today
            }).sort("created_at", -1)
        )

        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "Attendance"

        sheet.append([
            "Name",
            "ID",
            "Role",
            "Level/Dept",
            "Date",
            "Time",
            "Status",
        ])

        for record in records:
            sheet.append([
                record.get("person_name", ""),
                record.get("student_id", ""),
                record.get("role", ""),
                record.get("level", ""),
                record.get("date", ""),
                record.get("time", ""),
                record.get("status", ""),
            ])

        excel_file = io.BytesIO()
        workbook.save(excel_file)
        excel_file.seek(0)

        filename = f"attendance_{today}.xlsx"

        return send_file(
            excel_file,
            as_attachment=True,
            download_name=filename,
            mimetype=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
        )

    except PyMongoError:
        app.logger.exception("Error exporting attendance.")
        flash("Could not export attendance records.", "error")
        return redirect(url_for("list_records"))


# ================= Error Handlers =================

@app.errorhandler(413)
def file_too_large(error):
    flash("The uploaded file is too large. Maximum size is 5 MB.", "error")
    return redirect(request.referrer or url_for("add_person"))


# ================= Run Application =================

if __name__ == "__main__":
    host = os.environ.get(
        "HOST",
        "127.0.0.1",
    )

    port = int(os.environ.get("PORT", "5000"))

    app.run(
        debug=False,
        host=host,
        port=port,
    )

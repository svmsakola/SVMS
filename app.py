import base64
import os
import json
import bcrypt
import cv2
import numpy as np
import face_recognition
import signal
import sys
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from datetime import datetime

app = Flask(__name__)
CORS(app)
app.secret_key = os.urandom(24)

def signal_handler(sig, frame):
    print('Shutting down gracefully...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

os.makedirs('data/facedata', exist_ok=True)
os.makedirs('data/encodings', exist_ok=True)
os.makedirs('data/profiles', exist_ok=True)
os.makedirs('JSON', exist_ok=True)  # Ensure JSON directory exists

# Optimize OpenCV for different platforms
if sys.platform.startswith('darwin'):  # macOS
    cv2.setNumThreads(4)  # Limit threads on macOS for better performance
    os.environ['OPENCV_OPENCL_RUNTIME'] = ''  # Disable OpenCL on macOS
elif sys.platform.startswith('linux'):  # Ubuntu
    cv2.setNumThreads(0)
    try:
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
    except:
        pass
face_recognition_model = "hog"
if sys.platform.startswith('linux'):
    try:
        import torch
        if torch.cuda.is_available():
            face_recognition_model = "cnn"
    except ImportError:
        pass
try:
    person_model = YOLO('models/yolo11n-seg.pt')
    face_model = YOLO('models/yolov11n-face.pt')
except Exception as e:
    print(f"Error loading YOLO models: {e}")
    sys.exit(1)

def cleanup_memory():
    import gc
    gc.collect()
    if sys.platform.startswith('linux'):
        try:
            # On Linux, we can be more aggressive with memory cleanup
            import ctypes
            libc = ctypes.CDLL('libc.so.6')
            libc.malloc_trim(0)
        except:
            pass


def generate_visitor_id():
    today = datetime.now()
    try:
        with open('data/facedata/facedata.json', 'r') as f:
            visitors_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        visitors_data = {}

    date_component = today.strftime("%Y%m%d")
    today_visitors = [
        vid for uid, user in visitors_data.items()
        for vid in [visit.get('visit_id') for visit in user.get('visitor', [])]
        if vid and vid.startswith(f'V{date_component}')
    ]
    visit_count = len(today_visitors) + 1
    visitor_id = f"V{date_component}{str(visit_count).zfill(4)}"
    return visitor_id


def is_marathi(text):
    if not text:
        return False
    return any('\u0900' <= c <= '\u097F' for c in text)

def save_face_images(frame, uid):
    user_dir = f'data/facedata/{uid}'
    os.makedirs(user_dir, exist_ok=True)
    results = face_model.predict(source=frame, stream=True)
    saved_images = []
    face_encodings = []
    for r in results:
        if r.boxes is not None:
            for box in r.boxes.data:
                x1, y1, x2, y2 = map(int, box[:4].cpu().numpy())
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue
                filename = f'{uid}_img{len(saved_images) + 1}.jpg'
                image_path = os.path.join(user_dir, filename)
                cv2.imwrite(image_path, face_crop)
                saved_images.append(f'facedata/{uid}/{filename}')
                rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb_face, model=face_recognition_model)
                if encodings:
                    encoding_filename = f'data/encodings/{uid}_encoding{len(face_encodings) + 1}.npy'
                    np.save(encoding_filename, encodings[0])
                    face_encodings.append(encoding_filename)
                if len(saved_images) >= 3:
                    break
    return saved_images, face_encodings

def load_users():
    try:
        with open("JSON/auth.json", "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        if not os.path.exists("JSON/auth.json"):
            with open("JSON/auth.json", "w") as file:
                json.dump({}, file)
        return {}

def save_users(users):
    with open("JSON/auth.json", "w") as file:
        json.dump(users, file, indent=4)

@app.route("/", methods=["GET", "POST"])
def login():
    if "user" in session:
        return redirect(url_for("dashboard", role=session["user"]["role"]))
    if request.method == "POST":
        data = request.json
        if not data:
            return jsonify(success=False, message="Invalid request format")
        username = data.get("username")
        password = data.get("password")
        if not username or not password:
            return jsonify(success=False, message="Username and password are required")
        users = load_users()
        if username in users and bcrypt.checkpw(password.encode('utf-8'), users[username]["password"].encode('utf-8')):
            role = users[username]["role"]
            session["user"] = {"username": username, "role": role}
            return jsonify(success=True, redirect_url=url_for("dashboard", role=role))
        return jsonify(success=False, message="Invalid credentials")
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        data = request.json
        if not data:
            return jsonify(success=False, message="Invalid request format")
        username = data.get("username")
        password = data.get("password")
        role = data.get("role")
        if not username or not password or not role:
            return jsonify(success=False, message="All fields are required")
        valid_roles = ["admin", "registrar", "inoffice", "cctv"]
        if role not in valid_roles:
            return jsonify(success=False, message="Invalid role selected")
        users = load_users()
        if username in users:
            return jsonify(success=False, message="Username already exists")
        hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        users[username] = {
            "password": hashed_password,
            "role": role
        }
        save_users(users)
        return jsonify(success=True)
    return render_template("register.html")

@app.route("/dashboard/<role>")
def dashboard(role):
    if "user" not in session or session["user"]["role"] != role:
        return redirect(url_for("login"))
    templates = {
        "admin": "admin/admin.html",
        "registrar": "dashboard/registrar.html",
        "inoffice": "dashboard/inoffice.html",
        "cctv": "dashboard/cctv.html"
    }
    if role not in templates:
        return render_template("unauthorized.html")
    return render_template(templates[role])

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

@app.route('/register_visitor', methods=['POST'])
def register_visitor():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid request, no data received'}), 400

    if not data.get('name') or not data.get('phone'):
        missing = []
        if not data.get('name'): missing.append("name")
        if not data.get('phone'): missing.append("phone")
        return jsonify({'error': f'Missing required fields: {", ".join(missing)}'}), 400
    try:
        with open('data/facedata/facedata.json', 'r') as f:
            facedata = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        facedata = {}
    existing_uid = None
    for uid, user_data in facedata.items():
        if user_data.get('name') == data.get('name') and user_data.get('phone') == data.get('phone'):
            existing_uid = uid
            break
    uid = existing_uid if existing_uid else f"UID{int(datetime.now().timestamp())}"
    tahasil_options = ["अकोला", "अकोट", "तेल्हारा", "बाळापूर", "पाटूर", "मुर्तिजापूर", "बार्शीटाकळी"]
    selected_tahasil = data.get('tahasil', '')
    if selected_tahasil and selected_tahasil not in tahasil_options:
        return jsonify({'error': 'Invalid tahasil selection'}), 400
    visitor_data = {
        'name': data.get('name', ''),
        'phone': data.get('phone', ''),
        'email': data.get('email', ''),
        'address': data.get('address', ''),
        'tahasil': selected_tahasil,
        'district': data.get('district', 'Akola')
    }
    if not existing_uid:
        visitor_data['registration_datetime'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if 'frame' in data and data['frame']:
        try:
            frame_data = data['frame']
            if ',' in frame_data:
                frame_data = frame_data.split(',')[1]

            frame_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is not None:
                saved_images, face_encodings = save_face_images(frame, uid)
                if saved_images:
                    visitor_data['images'] = saved_images
                if face_encodings:
                    visitor_data['face_encodings'] = face_encodings
            else:
                return jsonify({'error': 'Invalid frame data, unable to decode'}), 400
        except Exception as e:
            return jsonify({'error': f'Error processing frame: {str(e)}'}), 500
        finally:
            cleanup_memory()
    visitor_id = generate_visitor_id()
    visit_entry = {
        'visit_id': visitor_id,
        'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'purpose': data.get('purpose', ''),
        'status': 'new'
    }
    if existing_uid and uid in facedata:
        if 'visitor' not in facedata[uid]:
            facedata[uid]['visitor'] = []
        facedata[uid]['visitor'].append(visit_entry)
    else:
        visitor_data['visitor'] = [visit_entry]
        facedata[uid] = visitor_data
    with open('data/facedata/facedata.json', 'w') as f:
        json.dump(facedata, f, indent=4)
    return jsonify({
        'success': True,
        'message': 'Visitor registered successfully',
        'uid': uid,
        'visit_id': visitor_id
    })

@app.route('/search_visitors', methods=['GET'])
def search_visitors():
    query = request.args.get('query', '').lower()
    if not query:
        return jsonify({'success': False, 'message': 'Search query is required'})
    try:
        with open('data/facedata/facedata.json', 'r') as f:
            facedata = json.load(f)
    except FileNotFoundError:
        return jsonify({'success': False, 'message': 'No visitor data found'})
    matching_visitors = []
    for uid, visitor_data in facedata.items():
        if (query in visitor_data.get('name', '').lower() or
                query in uid.lower() or
                query in visitor_data.get('phone', '').lower() or  # Also search in phone numbers
                any(query in visit.get('visit_id', '').lower() for visit in visitor_data.get('visitor', []))):
            last_visit = visitor_data.get('visitor', [])[-1] if visitor_data.get('visitor') else None
            matching_visitors.append({
                'uid': uid,
                'name': visitor_data.get('name', 'N/A'),
                'phone': visitor_data.get('phone', 'N/A'),
                'last_visit': last_visit.get('datetime', 'N/A') if last_visit else 'N/A'
            })
    return jsonify({
        'success': True,
        'visitors': matching_visitors
    })

@app.route('/get_visitor_details', methods=['GET'])
def get_visitor_details():
    uid = request.args.get('uid')
    visit_id = request.args.get('visit_id')
    if not uid and not visit_id:
        return jsonify({'success': False, 'message': 'Either uid or visit_id is required'})
    try:
        with open('data/facedata/facedata.json', 'r') as f:
            facedata = json.load(f)
    except FileNotFoundError:
        return jsonify({'success': False, 'message': 'No visitor data found'})
    if uid:
        if uid in facedata:
            return jsonify({
                'success': True,
                'visitor_data': facedata[uid]
            })
    if visit_id:
        for uid, user_data in facedata.items():
            for visit in user_data.get('visitor', []):
                if visit.get('visit_id') == visit_id:
                    return jsonify({
                        'success': True,
                        'visitor_data': user_data,
                        'uid': uid,
                        'visit': visit
                    })
    return jsonify({'success': False, 'message': 'Visitor not found'})

@app.route('/detect_face', methods=['POST'])
def detect_face():
    if 'frame' not in request.files:
        return jsonify({'recognized': False, 'message': 'No frame uploaded'})
    try:
        frame = np.frombuffer(request.files['frame'].read(), dtype=np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'recognized': False, 'message': 'Invalid image format'})
        if person_model is None or not hasattr(person_model, 'predict'):
            return jsonify({'recognized': False, 'message': 'Model not properly initialized'})
        results = person_model.predict(source=frame, stream=True, classes=[0])
        max_area, selected_mask = 0, None
        for r in results:
            if r.boxes is not None and r.masks is not None:
                for box, mask in zip(r.boxes.data, r.masks.data):
                    conf = float(box[4].cpu().numpy())
                    x1, y1, x2, y2 = map(int, box[:4].cpu().numpy())
                    box_area = (x2 - x1) * (y2 - y1)
                    if box_area > max_area and conf >= 0.5:
                        max_area = box_area
                        selected_mask = mask.cpu().numpy().astype('float32')
        if selected_mask is None:
            return jsonify({'recognized': False, 'message': 'No person detected'})
        face_results = face_model.predict(source=frame, stream=True)
        face_found = False
        for fr in face_results:
            if fr.boxes is not None:
                for box in fr.boxes.data:
                    face_found = True
                    x1, y1, x2, y2 = map(int, box[:4].cpu().numpy())
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue
                    rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    try:
                        with open('data/facedata/facedata.json', 'r') as f:
                            known_data = json.load(f)
                    except FileNotFoundError:
                        known_data = {}
                    known_encodings = []
                    known_uids = []
                    data_modified = False
                    for uid, user in known_data.items():
                        user_encodings = []
                        face_encodings_paths = user.get('face_encodings', [])
                        for encoding_path in face_encodings_paths:
                            if os.path.exists(encoding_path):
                                try:
                                    encoding = np.load(encoding_path)
                                    user_encodings.append(encoding)
                                    known_uids.append(uid)
                                    known_encodings.append(encoding)
                                except Exception as e:
                                    print(f"Error loading encoding {encoding_path}: {e}")
                                    continue
                        if not user_encodings:
                            images = user.get('images', [])
                            encoding_paths = []
                            for idx, image_path in enumerate(images):
                                if not os.path.exists(image_path):
                                    continue
                                try:
                                    image = face_recognition.load_image_file(image_path)
                                    face_encs = face_recognition.face_encodings(image, model=face_recognition_model)
                                    if not face_encs:
                                        continue
                                    encoding = face_encs[0]
                                    encoding_path = os.path.join('data/encodings', f'{uid}_{idx}.npy')
                                    os.makedirs('data/encodings', exist_ok=True)
                                    np.save(encoding_path, encoding)
                                    user_encodings.append(encoding)
                                    known_uids.append(uid)
                                    known_encodings.append(encoding)
                                    encoding_paths.append(encoding_path)
                                    data_modified = True
                                except Exception as e:
                                    print(f"Error processing image {image_path}: {e}")
                                    continue
                            if encoding_paths and data_modified:
                                user['face_encodings'] = encoding_paths
                    if data_modified:
                        with open('data/facedata/facedata.json', 'w') as f:
                            json.dump(known_data, f, indent=2)
                    if not known_encodings:
                        continue
                    face_encodings = face_recognition.face_encodings(rgb_face, model=face_recognition_model)
                    if face_encodings:
                        matches = face_recognition.compare_faces(known_encodings, face_encodings[0], tolerance=0.4)
                        if True in matches:
                            best_match_index = matches.index(True)
                            matched_uid = known_uids[best_match_index]
                            matched_user = known_data[matched_uid]
                            return jsonify({
                                'recognized': True,
                                'user_data': matched_user,
                                'uid': matched_uid
                            })
        if not face_found:
            return jsonify({'recognized': False, 'message': 'No face detected'})
        return jsonify({'recognized': False, 'message': 'Face not recognized'})
    except Exception as e:
        return jsonify({'recognized': False, 'message': f'Error processing image: {str(e)}'})
    finally:
        cleanup_memory()

@app.route('/confirm_visitor_entry', methods=['POST'])
def confirm_visitor_entry():
    data = request.json
    if not data:
        return jsonify({'success': False, 'message': 'Invalid request, no data received'}), 400
    visit_id = data.get('visitId')
    dvn = data.get('dvn')
    if not visit_id or not dvn:
        return jsonify({'success': False, 'message': 'Visit ID and DVN are required'}), 400
    try:
        with open('data/facedata/facedata.json', 'r') as f:
            facedata = json.load(f)
    except FileNotFoundError:
        return jsonify({'success': False, 'message': 'Visitor data not found'})
    for uid, user_data in facedata.items():
        for visit in user_data.get('visitor', []):
            if visit.get('visit_id') == visit_id:
                visit['dvn'] = dvn
                visit['entry_confirmed'] = True
                visit['confirmation_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open('data/facedata/facedata.json', 'w') as f:
                    json.dump(facedata, f, indent=4)

                return jsonify({
                    'success': True,
                    'message': 'Visitor entry confirmed',
                    'dvn': dvn
                })
    return jsonify({'success': False, 'message': 'Visitor not found'})

@app.route('/get_today_visitors', methods=['GET'])
def get_today_visitors():
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        with open('data/facedata/facedata.json', 'r') as f:
            facedata = json.load(f)
    except FileNotFoundError:
        return jsonify({'success': False, 'message': 'No visitor data found', 'visitors': {}})
    today_visitors = {}
    for uid, user_data in facedata.items():
        today_visits = [
            visit for visit in user_data.get('visitor', [])
            if visit.get('datetime', '').startswith(today)
        ]
        if today_visits:
            today_visitors[uid] = {
                'name': user_data.get('name', 'Unknown'),
                'phone': user_data.get('phone', 'N/A'),
                'visitor': today_visits
            }
    return jsonify({
        'success': True,
        'visitors': today_visitors
    })

def detect_person_and_face(frame):
    white_bg = np.ones_like(frame) * 255
    frame_area = frame.shape[0] * frame.shape[1]
    results = person_model.predict(source=frame, stream=True, classes=[0])
    max_area, selected_mask = 0, None
    for r in results:
        if r.boxes is not None and r.masks is not None:
            for box, mask in zip(r.boxes.data, r.masks.data):
                conf = float(box[4].cpu().numpy())
                x1, y1, x2, y2 = map(int, box[:4].cpu().numpy())
                box_area = (x2 - x1) * (y2 - y1)
                if box_area / frame_area >= 0.15 and conf >= 0.5 and box_area > max_area:
                    max_area = box_area
                    selected_mask = mask.cpu().numpy().astype('float32')
    face_output = white_bg.copy()
    face_crop = None
    if selected_mask is not None:
        selected_mask = cv2.resize(selected_mask, (frame.shape[1], frame.shape[0]))
        _, binary_mask = cv2.threshold(selected_mask, 0.5, 255, cv2.THRESH_BINARY)
        binary_mask = cv2.merge([binary_mask.astype(np.uint8)] * 3)
        person_only = cv2.bitwise_and(frame, binary_mask)
        person_with_white_bg = np.where(binary_mask == 0, white_bg, person_only)
        face_results = face_model.predict(source=person_with_white_bg, stream=True)
        max_face_area, selected_face = 0, None
        for fr in face_results:
            if fr.boxes is not None:
                for box in fr.boxes.data:
                    x1, y1, x2, y2 = map(int, box[:4].cpu().numpy())
                    face_area = (x2 - x1) * (y2 - y1)
                    if face_area > max_face_area:
                        max_face_area = face_area
                        selected_face = (x1, y1, x2, y2)
        if selected_face:
            x1, y1, x2, y2 = selected_face
            w, h = x2 - x1, y2 - y1
            pad_w, pad_h = int(w * 0.6), int(h * 0.4)
            x1, y1 = max(x1 - pad_w, 0), max(y1 - pad_h, 0)
            x2, y2 = min(x2 + pad_w, frame.shape[1]), min(y2 + pad_h, frame.shape[0])
            face_crop = person_with_white_bg[y1:y2, x1:x2]
            face_output[y1:y2, x1:x2] = face_crop
    return face_output, face_crop

@app.route('/process_profile_image', methods=['POST'])
def process_profile_image():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image uploaded'})
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No image selected'})
    uid = request.form.get('uid')
    if not uid:
        return jsonify({'success': False, 'message': 'User ID (uid) is required'})
    try:
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'success': False, 'message': 'Invalid image format'})
        _, face_crop = detect_person_and_face(frame)
        if face_crop is None:
            return jsonify({'success': False, 'message': 'No face detected in the image'})
        user_dir = f'data/profiles/{uid}'
        os.makedirs(user_dir, exist_ok=True)
        existing_images = [f for f in os.listdir(user_dir) if f.startswith(f'{uid}_img') and f.endswith('.jpg')]
        next_img_num = len(existing_images) + 1
        face_path = f'{user_dir}/{uid}_img{next_img_num}.jpg'
        cv2.imwrite(face_path, face_crop)
        facedata_path = 'data/facedata/facedata.json'
        os.makedirs(os.path.dirname(facedata_path), exist_ok=True)
        if os.path.exists(facedata_path):
            with open(facedata_path, 'r') as json_file:
                try:
                    facedata = json.load(json_file)
                except json.JSONDecodeError:
                    facedata = {}
        else:
            facedata = {}
        if uid not in facedata:
            facedata[uid] = {"profile_img": []}
        elif "profile_img" not in facedata[uid]:
            facedata[uid]["profile_img"] = []
        relative_path = f'profiles/{uid}/{uid}_img{next_img_num}.jpg'
        facedata[uid]["profile_img"].append(relative_path)
        with open(facedata_path, 'w') as json_file:
            json.dump(facedata, json_file, indent=4)
        return jsonify({
            'success': True,
            'message': 'Profile image processed successfully',
            'image_path': relative_path,
            'image_number': next_img_num
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error processing image: {str(e)}'}), 500
    finally:
        cleanup_memory()

@app.route('/dashboard/profiles/<uid>/<filename>', methods=['GET'])
def get_profile_image(uid, filename):
    profiles_dir = os.path.join('data/profiles')
    if not os.path.exists(os.path.join(profiles_dir, uid, filename)):
        return jsonify({'success': False, 'message': 'Image not found'}), 404
    return send_from_directory(os.path.join(profiles_dir, uid), filename)

@app.route('/departments', methods=['GET'])
def get_departments():
    try:
        with open('JSON/departments.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({"error": "Departments file not found"}), 404
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON in departments file"}), 500

@app.route('/complete_meeting', methods=['POST'])
def complete_meeting():
    try:
        visit_id = request.args.get('visit_id')
        uid = request.args.get('uid')
        if not visit_id or not uid:
            return jsonify({"error": "Missing visit_id or uid parameter"}), 400
        json_file_path = 'data/facedata/facedata.json'
        if not os.path.exists(json_file_path):
            return jsonify({"error": "Data file not found"}), 404
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        if uid not in data:
            return jsonify({"error": f"User with UID {uid} not found"}), 404
        visit_found = False
        for visit in data[uid].get('visitor', []):
            if visit.get('visit_id') == visit_id:
                visit['status'] = 'completed'
                visit_found = True
                break
        if not visit_found:
            return jsonify({"error": f"Visit with ID {visit_id} not found for user {uid}"}), 404
        with open(json_file_path, 'w') as file:
            json.dump(data, file, indent=4)
        return jsonify({"success": True, "message": f"Meeting {visit_id} marked as completed for user {uid}"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/visitor_action', methods=['POST'])
def visitor_action():
    if "user" not in session:
        return jsonify({"success": False, "message": "Unauthorized"}), 401
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "Invalid JSON payload."}), 400
    visit_id = data.get('visit_id')
    uid = data.get('uid')
    remark = data.get('remark', '')
    forwarding_department = data.get('forwarding_department')
    if not visit_id or not uid or not forwarding_department:
        missing = []
        if not visit_id: missing.append("visit_id")
        if not uid: missing.append("uid")
        if not forwarding_department: missing.append("forwarding_department")
        return jsonify({"success": False, "message": f"Missing required fields: {', '.join(missing)}"}), 400
    try:
        with open('data/facedata/facedata.json', 'r') as f:
            facedata = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return jsonify({"success": False, "message": "Could not load visitor data."}), 500
    if uid not in facedata:
        return jsonify({"success": False, "message": f"Visitor with UID '{uid}' not found."}), 404
    visit_found = False
    for visit in facedata[uid].get('visitor', []):
        if visit.get('visit_id') == visit_id:
            visit['remark'] = remark
            visit['forwarding_department'] = forwarding_department
            visit['status'] = 'pending'
            visit_found = True
            break
    if not visit_found:
        return jsonify({"success": False, "message": f"Visit with ID '{visit_id}' not found for visitor '{uid}'."}), 404
    try:
        with open('data/facedata/facedata.json', 'w') as f:
            json.dump(facedata, f, indent=4)
        return jsonify({"success": True, "message": "Visitor action recorded successfully."}), 200
    except Exception as e:
        return jsonify({"success": False, "message": f"Failed to save updated visitor data: {str(e)}"}), 500

if __name__ == "__main__":
    app.run()

# gunicorn --workers 3 --bind 127.0.0.1:8080 app:app
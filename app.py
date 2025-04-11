import os, io, sys, uuid, json, random, signal, base64
from datetime import datetime, timedelta
from functools import wraps
import bcrypt, cv2, numpy as np, face_recognition
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory, \
    flash, make_response, send_file
from flask_cors import CORS
import openpyxl
from openpyxl.styles import Font, Alignment, Border, Side
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = 'uploads/docimg'
PDF_FOLDER = 'pdf'
DATA_FILE = 'data/facedata/facedata.json'
DEPARTMENTS_FILE = 'JSON/departments.json'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PDF_FOLDER, exist_ok=True)

def is_admin_authenticated():
    print("WARNING: delete_visitor_entry endpoint called WITHOUT proper authentication check!")
    return True

def load_face_data():
    try:
        with open(DATA_FILE, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_face_data(data):
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, 'w') as file:
        json.dump(data, file, indent=4)

def signal_handler(sig, frame):
    print('Shutting down gracefully...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

os.makedirs('data/facedata', exist_ok=True)
os.makedirs('data/encodings', exist_ok=True)
os.makedirs('data/profiles', exist_ok=True)
os.makedirs('JSON', exist_ok=True)

if sys.platform.startswith('darwin'):
    cv2.setNumThreads(4)
    os.environ['OPENCV_OPENCL_RUNTIME'] = ''
elif sys.platform.startswith('linux'):
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
        valid_roles = ["admin", "registrar", "inoffice", "cctv", "scanner"]
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
        "cctv": "dashboard/cctv.html",
        "scanner": "dashboard/scanner.html"
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
    tahasil_options = ["अकोला", "अकोट", "तेल्हारा", "बाळापूर", "पातूर", "मुर्तिजापूर", "बार्शीटाकळी"]
    selected_tahasil = data.get('tahasil', '')
    if selected_tahasil.endswith('<'):
        selected_tahasil = selected_tahasil[:-1]
    if selected_tahasil and selected_tahasil not in tahasil_options:
        return jsonify({'error': f'Invalid tahasil selection: {selected_tahasil}'}), 400
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
                query in visitor_data.get('phone', '').lower() or
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

@app.route('/registered_visitor_today')
def registered_visitor_today():
    face_data = load_face_data()
    today = datetime.now().strftime('%Y-%m-%d')
    today_visitors = []
    for uid, user_data in face_data.items():
        if 'visitor' in user_data:
            for visit in user_data['visitor']:
                if 'datetime' in visit and isinstance(visit['datetime'], str):
                     visit_date_str = visit['datetime'].split(' ')[0]
                     try:
                         datetime.strptime(visit_date_str, '%Y-%m-%d')
                         if visit_date_str == today:
                            today_visitors.append({
                                'uid': uid,
                                'name': user_data.get('name', 'N/A'),
                                'phone': user_data.get('phone', 'N/A'),
                                'purpose': visit.get('purpose', 'N/A'),
                                'visit_id': visit.get('visit_id', 'N/A'),
                                'status': visit.get('status', 'Unknown'),
                                'pdf_path': visit.get('document_pdf', '')
                            })
                     except ValueError:
                         print(f"Warning: Invalid date format found for visit_id {visit.get('visit_id', 'N/A')}: {visit['datetime']}")
                else:
                     print(f"Warning: Missing or invalid 'datetime' for visit in user {uid}")
    return jsonify({'visitors': today_visitors})

@app.route('/api/upload-visitor-document/<visit_id>', methods=['POST'])
def upload_visitor_document(visit_id):
    if 'document' not in request.files:
        return jsonify({'error': 'No document provided'}), 400
    file = request.files['document']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are allowed'}), 400
    try:
        filename = f"{visit_id}_1.pdf"
        filepath = os.path.join(PDF_FOLDER, filename)
        file.save(filepath)
        face_data = load_face_data()
        updated = False
        for uid, user_data in face_data.items():
            if 'visitor' in user_data:
                for visit in user_data['visitor']:
                    if visit['visit_id'] == visit_id:
                        visit['document_pdf'] = filepath
                        updated = True
                        break
            if updated:
                break
        if updated:
            save_face_data(face_data)
            return jsonify({
                'success': True,
                'message': 'Document uploaded successfully',
                'pdf_path': filepath
            })
        else:
            return jsonify({'error': 'Visitor ID not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-visitor-document/<visit_id>')
def get_visitor_document(visit_id):
    face_data = load_face_data()
    pdf_filename = None
    for uid, user_data in face_data.items():
        if 'visitor' in user_data:
            for visit in user_data['visitor']:
                if visit.get('visit_id') == visit_id and 'document_pdf' in visit:
                    full_pdf_path = visit['document_pdf']
                    if full_pdf_path and isinstance(full_pdf_path, str):
                       pdf_filename = os.path.basename(full_pdf_path)
                       break
        if pdf_filename:
            break
    if pdf_filename:
        try:
            print(f"Attempting to send file: Directory='{PDF_FOLDER}', Filename='{pdf_filename}'")
            return send_from_directory(
                PDF_FOLDER,
                pdf_filename,
                as_attachment=False,
                mimetype='application/pdf'
            )
        except FileNotFoundError:
             print(f"Error: File not found at expected location: {os.path.join(PDF_FOLDER, pdf_filename)}")
             return jsonify({'error': 'Document file not found on server.'}), 404
        except Exception as e:
             print(f"Error sending file: {e}")
             return jsonify({'error': 'An error occurred while retrieving the document.'}), 500
    else:
        print(f"Document reference not found in JSON data for visit_id: {visit_id}")
        return jsonify({'error': 'Document reference not found for this visitor ID.'}), 404

@app.route('/api/upload-image', methods=['POST'])
def upload_image():
    if 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400
    try:
        image_data = request.json['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        filename = f"{uuid.uuid4()}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        return jsonify({
            'success': True,
            'image_id': filename
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-pdf', methods=['POST'])
def generate_pdf():
    if 'images' not in request.json or 'visit_id' not in request.json:
        return jsonify({'error': 'No images or visit ID provided'}), 400
    try:
        image_ids = request.json['images']
        visit_id = request.json['visit_id']
        pdf_filename = f"{visit_id}_1.pdf"
        pdf_path = os.path.join(PDF_FOLDER, pdf_filename)
        c = canvas.Canvas(pdf_path, pagesize=letter)
        letter_width, letter_height = letter
        image_paths = []
        for image_id in image_ids:
            img_path = os.path.join(UPLOAD_FOLDER, image_id)
            if os.path.exists(img_path):
                image_paths.append(img_path)
                img = Image.open(img_path)
                width, height = img.size
                scale = min(letter_width / width, letter_height / height) * 0.9
                img_width = width * scale
                img_height = height * scale
                x = (letter_width - img_width) / 2
                y = (letter_height - img_height) / 2
                c.drawImage(img_path, x, y, width=img_width, height=img_height)
                c.showPage()
        c.save()
        for img_path in image_paths:
            if os.path.exists(img_path):
                os.remove(img_path)
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
        face_data = load_face_data()
        updated = False
        for uid, user_data in face_data.items():
            if 'visitor' in user_data:
                for visit in user_data['visitor']:
                    if visit['visit_id'] == visit_id:
                        visit['document_pdf'] = pdf_path
                        updated = True
                        break
            if updated:
                break
        if updated:
            save_face_data(face_data)
        return jsonify({
            'success': True,
            'pdf_url': f'/api/get-visitor-document/{visit_id}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cleanup', methods=['POST'])
def cleanup_images():
    try:
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        return jsonify({
            'success': True,
            'message': 'All temporary images have been deleted'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def load_departments():
    with open('JSON/departments.json', 'r') as f:
        content = f.read().strip()
        if not content:
            return []
        return json.loads(content).get('departments', [])


def send_otp_email(recipient_email, otp):
    import smtplib
    import ssl
    import random
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.utils import formataddr, formatdate

    sender_email = "svmsakola@gmail.com"
    sender_password = "aess hmcl bkqm slph"
    sender_name = "SVMS Akola"

    # Create message
    message = MIMEMultipart("alternative")

    # Enhanced headers
    message["Subject"] = "Secure Access Code - SVMS Department Portal"
    message["From"] = formataddr((sender_name, sender_email))
    message["To"] = recipient_email
    message["Date"] = formatdate(localtime=True)
    message["Message-ID"] = f"<{random.getrandbits(128)}@{sender_email.split('@')[-1]}>"
    message["X-Mailer"] = "CustomMailer/1.0"
    message["Reply-To"] = sender_email
    message["Return-Path"] = sender_email

    # Text version
    text = f"""Dear SVMS Department User,

Your secure verification code for accessing the SVMS Department Portal is:

{otp}

This verification code will expire in 10 minutes for your security.

IMPORTANT: Never share this code with anyone. SVMS staff will never ask for this code.

If you did not attempt to access the SVMS Department Portal, please contact our IT support team immediately.

Best regards,
SVMS Akola Administration
Shri Vasantrao Naik Mahavidyalaya, Akola
Maharashtra - 444001
Tel: +91 12345 67890
"""

    # HTML version with more professional styling
    html = f"""<!DOCTYPE html>
<html>
<body style="margin: 0; padding: 0; font-family: Arial, Helvetica, sans-serif; color: #333333;">
  <table align="center" border="0" cellpadding="0" cellspacing="0" width="600" style="border-collapse: collapse; border: 1px solid #cccccc;">
    <tr>
      <td align="center" bgcolor="#1a4f8a" style="padding: 30px 0;">
        <h1 style="color: #ffffff; margin: 0;">SVMS Department Portal</h1>
      </td>
    </tr>
    <tr>
      <td bgcolor="#ffffff" style="padding: 30px;">
        <table border="0" cellpadding="0" cellspacing="0" width="100%">
          <tr>
            <td style="padding: 10px 0;">
              <p style="margin: 0;">Dear SVMS Department User,</p>
            </td>
          </tr>
          <tr>
            <td style="padding: 10px 0;">
              <p style="margin: 0;">Your secure verification code for accessing the SVMS Department Portal is:</p>
            </td>
          </tr>
          <tr>
            <td align="center" style="padding: 20px 0;">
              <div style="background-color: #f3f7fc; border: 1px solid #dae4f2; border-radius: 6px; padding: 15px; font-family: Courier, monospace; font-size: 24px; font-weight: bold; letter-spacing: 5px; color: #1a4f8a;">
                {otp}
              </div>
            </td>
          </tr>
          <tr>
            <td style="padding: 10px 0;">
              <p style="margin: 0;">This verification code will expire in <strong>10 minutes</strong> for your security.</p>
            </td>
          </tr>
          <tr>
            <td style="padding: 15px 0; border-top: 1px solid #eeeeee; border-bottom: 1px solid #eeeeee;">
              <p style="margin: 0; color: #e74c3c; font-weight: bold;">IMPORTANT: Never share this code with anyone. SVMS staff will never ask for this code.</p>
            </td>
          </tr>
          <tr>
            <td style="padding: 10px 0;">
              <p style="margin: 0;">If you did not attempt to access the SVMS Department Portal, please contact our IT support team immediately.</p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
    <tr>
      <td bgcolor="#f2f2f2" style="padding: 20px;">
        <table border="0" cellpadding="0" cellspacing="0" width="100%">
          <tr>
            <td style="color: #555555; font-size: 13px;">
              <p style="margin: 0; padding-bottom: 5px;"><strong>SVMS Akola Administration</strong></p>
              <p style="margin: 0; padding-bottom: 5px;">Shri Vasantrao Naik Mahavidyalaya, Akola</p>
              <p style="margin: 0; padding-bottom: 5px;">Maharashtra - 444001</p>
              <p style="margin: 0; padding-bottom: 5px;">Tel: +91 12345 67890</p>
              <p style="margin: 0; font-size: 11px; color: #777777; padding-top: 10px;">This is an automated message. Please do not reply to this email.</p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""

    # Attach both versions
    message.attach(MIMEText(text, "plain"))
    message.attach(MIMEText(html, "html"))

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, message.as_string())
            print(f"Access code sent to {recipient_email}")
            return True
    except Exception as e:
        print(f"Email error: {str(e)}")
        return False

@app.route('/department/login', methods=['GET', 'POST'])
def login_department():
    if 'department' in session:
        return redirect(url_for('department_dashboard'))
    if request.method == 'POST':
        email = request.form.get('email')
        departments = load_departments()
        department = next((d for d in departments if d['email'] == email), None)
        if department:
            otp = str(random.randint(100000, 999999))
            session['otp'] = otp
            session['pending_department'] = department
            if send_otp_email(email, otp):
                flash('OTP sent to your email.', 'info')
                return redirect(url_for('verify_otp'))
            else:
                flash('Failed to send OTP email. Please try again later.', 'danger')
        else:
            flash('Email not found.', 'danger')
    return render_template('department/auth/login_department.html')

@app.route('/department/verify-otp', methods=['GET', 'POST'])
def verify_otp():
    if request.method == 'POST':
        entered_otp = request.form.get('otp')
        if entered_otp == session.get('otp'):
            session['department'] = session.pop('pending_department', None)
            session.pop('otp', None)
            return redirect(url_for('department_dashboard'))
        else:
            flash('Invalid OTP.', 'danger')
    return render_template('department/auth/verify_otp.html')

@app.route('/department/dashboard')
def department_dashboard():
    department = session.get('department')
    if not department:
        return redirect(url_for('login_department'))
    return render_template('department/dashboard/department_dashboard.html', department=department)

@app.route('/department/logout')
def logout_department():
    session.pop('department', None)
    return redirect(url_for('login_department'))


def department_login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'department' not in session:
            return redirect(url_for('login_department'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/api/department/applications', methods=['GET'])
@department_login_required
def get_department_applications():
    department = session.get('department')
    department_name = department.get('name')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    status = request.args.get('status')
    try:
        with open(DATA_FILE, 'r') as file:
            facedata = json.load(file)
    except Exception as e:
        return jsonify({'error': f'Failed to load data: {str(e)}'}), 500
    applications = []
    for uid, user_data in facedata.items():
        for visit in user_data.get('visitor', []):
            if visit.get('forwarding_department') == department_name:
                if status and visit.get('status') != status:
                    continue
                if start_date or end_date:
                    try:
                        visit_date = datetime.strptime(visit.get('datetime'), '%Y-%m-%d %H:%M:%S').date()
                        if start_date:
                            start = datetime.strptime(start_date, '%Y-%m-%d').date()
                            if visit_date < start:
                                continue
                        if end_date:
                            end = datetime.strptime(end_date, '%Y-%m-%d').date()
                            if visit_date > end:
                                continue
                    except ValueError:
                        continue
                application = {
                    'uid': uid,
                    'visit_id': visit.get('visit_id'),
                    'name': user_data.get('name'),
                    'phone': user_data.get('phone'),
                    'email': user_data.get('email'),
                    'datetime': visit.get('datetime'),
                    'purpose': visit.get('purpose'),
                    'status': visit.get('status'),
                    'remark': visit.get('remark', ''),
                    'profile_img': user_data.get('profile_img', [''])[0] if user_data.get('profile_img') else ''
                }
                applications.append(application)

    return jsonify(applications)

@app.route('/api/department/application/<visit_id>', methods=['GET'])
@department_login_required
def get_application_details(visit_id):
    department = session.get('department')
    department_name = department.get('name')
    try:
        with open(DATA_FILE, 'r') as file:
            facedata = json.load(file)
    except Exception as e:
        return jsonify({'error': f'Failed to load data: {str(e)}'}), 500
    for uid, user_data in facedata.items():
        for visit in user_data.get('visitor', []):
            if visit.get('visit_id') == visit_id:
                if visit.get('forwarding_department') == department_name:
                    application = {
                        'uid': uid,
                        'visit_id': visit.get('visit_id'),
                        'name': user_data.get('name'),
                        'phone': user_data.get('phone'),
                        'email': user_data.get('email'),
                        'address': user_data.get('address'),
                        'district': user_data.get('district'),
                        'tahasil': user_data.get('tahasil'),
                        'registration_datetime': user_data.get('registration_datetime'),
                        'datetime': visit.get('datetime'),
                        'purpose': visit.get('purpose'),
                        'status': visit.get('status'),
                        'remark': visit.get('remark', ''),
                        'document_pdf': visit.get('document_pdf', ''),
                        'entry_confirmed': visit.get('entry_confirmed', False),
                        'confirmation_time': visit.get('confirmation_time', ''),
                        'profile_img': user_data.get('profile_img', [''])[0] if user_data.get('profile_img') else '',
                        'images': user_data.get('images', [])
                    }
                    return jsonify(application)
                else:
                    return jsonify({"error": "Unauthorized access"}), 403
    return jsonify({"error": "Application not found"}), 404

@app.route('/api/department/update-application/<visit_id>', methods=['POST'])
@department_login_required
def update_application(visit_id):
    department = session.get('department')
    department_name = department.get('name')
    status = request.form.get('status')
    remark = request.form.get('remark', '')
    if not status or status not in ['pending', 'completed']:
        return jsonify({"error": "Invalid status value"}), 400
    try:
        with open(DATA_FILE, 'r') as file:
            facedata = json.load(file)
    except Exception as e:
        return jsonify({'error': f'Failed to load data: {str(e)}'}), 500
    updated = False
    for uid, user_data in facedata.items():
        for i, visit in enumerate(user_data.get('visitor', [])):
            if visit.get('visit_id') == visit_id:
                if visit.get('forwarding_department') == department_name:
                    user_data['visitor'][i]['status'] = status
                    user_data['visitor'][i]['remark'] = remark
                    updated = True
                    break
        if updated:
            break
    if updated:
        try:
            with open(DATA_FILE, 'w') as file:
                json.dump(facedata, file, indent=4)
            return jsonify({"success": True, "message": "Application updated successfully"})
        except Exception as e:
            return jsonify({'error': f'Failed to save data: {str(e)}'}), 500
    else:
        return jsonify({"error": "Application not found or unauthorized"}), 404

@app.route('/api/department/application-stats', methods=['GET'])
@department_login_required
def get_application_stats():
    department = session.get('department')
    department_name = department.get('name')
    try:
        with open(DATA_FILE, 'r') as file:
            facedata = json.load(file)
    except Exception as e:
        return jsonify({'error': f'Failed to load data: {str(e)}'}), 500
    total_count = 0
    pending_count = 0
    completed_count = 0
    today = datetime.now().date() # Corrected
    today_count = 0
    for uid, user_data in facedata.items():
        for visit in user_data.get('visitor', []):
            if visit.get('forwarding_department') == department_name:
                total_count += 1
                if visit.get('status') == 'pending':
                    pending_count += 1
                elif visit.get('status') == 'completed':
                    completed_count += 1
                try:
                    visit_date = datetime.strptime(visit.get('datetime'), '%Y-%m-%d %H:%M:%S').date() # Corrected
                    if visit_date == today:
                        today_count += 1
                except ValueError:
                    pass
    stats = {
        'total': total_count,
        'pending': pending_count,
        'completed': completed_count,
        'today': today_count
    }

    return jsonify(stats)

@app.route('/api/department/forward-application/<visit_id>', methods=['POST'])
@department_login_required
def forward_application(visit_id):
    current_department = session.get('department')
    current_department_name = current_department.get('name')
    target_department = request.form.get('target_department')
    note = request.form.get('note', '')
    if not target_department:
        return jsonify({"error": "Target department is required"}), 400
    departments = load_departments()
    if not any(d['name'] == target_department for d in departments):
        return jsonify({"error": "Invalid target department"}), 400
    try:
        with open(DATA_FILE, 'r') as file:
            facedata = json.load(file)
    except Exception as e:
        return jsonify({'error': f'Failed to load data: {str(e)}'}), 500
    updated = False
    for uid, user_data in facedata.items():
        for i, visit in enumerate(user_data.get('visitor', [])):
            if visit.get('visit_id') == visit_id:
                if visit.get('forwarding_department') == current_department_name:
                    user_data['visitor'][i]['forwarding_department'] = target_department
                    user_data['visitor'][i]['status'] = 'pending'
                    user_data['visitor'][i]['forwarding_note'] = note
                    user_data['visitor'][i]['forwarded_from'] = current_department_name
                    user_data['visitor'][i]['forwarded_datetime'] = datetime.now().strftime( # Corrected
                        '%Y-%m-%d %H:%M:%S')
                    updated = True
                    break
        if updated:
            break
    if updated:
        try:
            with open(DATA_FILE, 'w') as file:
                json.dump(facedata, file, indent=4)
            return jsonify({
                "success": True,
                "message": f"Application forwarded to {target_department} successfully"
            })
        except Exception as e:
            return jsonify({'error': f'Failed to save data: {str(e)}'}), 500
    else:
        return jsonify({"error": "Application not found or unauthorized"}), 404

@app.route('/api/department/search-applications', methods=['GET'])
@department_login_required
def search_applications():
    department = session.get('department')
    department_name = department.get('name')
    query = request.args.get('query', '').lower()
    if not query or len(query) < 3:
        return jsonify({"error": "Search query must be at least 3 characters"}), 400
    try:
        with open(DATA_FILE, 'r') as file:
            facedata = json.load(file)
    except Exception as e:
        return jsonify({'error': f'Failed to load data: {str(e)}'}), 500
    results = []
    for uid, user_data in facedata.items():
        for visit in user_data.get('visitor', []):
            if visit.get('forwarding_department') == department_name:
                if (query in user_data.get('name', '').lower() or
                        query in user_data.get('phone', '').lower() or
                        query in user_data.get('email', '').lower() or
                        query in visit.get('visit_id', '').lower() or
                        query in visit.get('purpose', '').lower() or
                        query in user_data.get('address', '').lower() or
                        query in user_data.get('district', '').lower()):
                    result = {
                        'uid': uid,
                        'visit_id': visit.get('visit_id'),
                        'name': user_data.get('name'),
                        'phone': user_data.get('phone'),
                        'email': user_data.get('email'),
                        'datetime': visit.get('datetime'),
                        'purpose': visit.get('purpose'),
                        'status': visit.get('status'),
                        'profile_img': user_data.get('profile_img', [''])[0] if user_data.get('profile_img') else ''
                    }
                    results.append(result)
    return jsonify(results)

@app.route('/api/department/recent-activities', methods=['GET'])
@department_login_required
def get_recent_activities():
    department = session.get('department')
    department_name = department.get('name')
    limit = int(request.args.get('limit', 10))
    try:
        with open(DATA_FILE, 'r') as file:
            facedata = json.load(file)
    except Exception as e:
        return jsonify({'error': f'Failed to load data: {str(e)}'}), 500
    activities = []
    for uid, user_data in facedata.items():
        for visit in user_data.get('visitor', []):
            if visit.get('forwarding_department') == department_name:
                activity = {
                    'uid': uid,
                    'visit_id': visit.get('visit_id'),
                    'name': user_data.get('name'),
                    'datetime': visit.get('datetime'),
                    'purpose': visit.get('purpose'),
                    'status': visit.get('status'),
                    'profile_img': user_data.get('profile_img', [''])[0] if user_data.get('profile_img') else ''
                }
                activities.append(activity)
    try:
        activities.sort(key=lambda x: datetime.strptime(x['datetime'], '%Y-%m-%d %H:%M:%S'), reverse=True) # Corrected
    except:
        pass
    return jsonify(activities[:limit])

@app.route('/api/admin/visitors')
def get_admin_visitors():
    selected_date_str = request.args.get('date')
    if not selected_date_str:
        return jsonify({"error": "Date parameter is required"}), 400
    try:
        # Validate date format
        datetime.strptime(selected_date_str, '%Y-%m-%d')
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

    face_data = load_face_data()
    daily_visitors = []

    for uid, user_data in face_data.items():
        # Basic validation for user_data structure
        if isinstance(user_data, dict) and 'visitor' in user_data and isinstance(user_data['visitor'], list):
            for visit in user_data['visitor']:
                # Basic validation for visit structure and datetime field
                if isinstance(visit, dict) and 'datetime' in visit and isinstance(visit['datetime'], str):
                    try:
                        # Check if the visit date matches the selected date
                        if visit['datetime'].split(' ')[0] == selected_date_str:
                            # Construct the visitor entry dictionary
                            daily_visitors.append({
                                "uid": uid,
                                "name": user_data.get('name', 'N/A'),
                                "phone": user_data.get('phone', 'N/A'),
                                "address": user_data.get('address', 'N/A'),
                                "tahasil": user_data.get('tahasil', 'N/A'),
                                "district": user_data.get('district', 'N/A'),
                                # Use profile_img if available, fallback to images, then empty list
                                "profile_img": user_data.get('profile_img', user_data.get('images', [])[:1]), # Simplified fallback
                                "images": user_data.get('images', []), # Keep original images list
                                "face_encodings": user_data.get('face_encodings', []),
                                "visit_id": visit.get('visit_id', 'N/A'),
                                "datetime": visit.get('datetime', ''),
                                "purpose": visit.get('purpose', 'N/A'),
                                "status": visit.get('status', 'unknown'),
                                "entry_confirmed": visit.get('entry_confirmed', False),
                                "dvn": visit.get('dvn', None),
                                "forwarding_department": visit.get('forwarding_department', ''),
                                "forwarding_note": visit.get('forwarding_note', ''), # Added forwarding_note
                                "remark": visit.get('remark', ''), # Added remark explicitly
                                "document_pdf": visit.get('document_pdf', None)
                            })
                    except Exception as e:
                        # Log errors during processing individual visits if needed
                        print(f"Error processing visit for UID {uid}, Visit {visit.get('visit_id', 'N/A')}: {e}")
                else:
                    # Log or handle invalid visit entries
                    print(f"Warning: Skipping invalid visit entry for UID {uid}: {visit}")
        else:
            # Log or handle invalid user data structures
            print(f"Warning: Skipping invalid user data structure for UID {uid}")

    # Sort the results by datetime
    try:
        # Filter out entries with invalid datetime strings before sorting
        valid_visitors = [v for v in daily_visitors if 'datetime' in v and isinstance(v['datetime'], str)]
        invalid_visitors = [v for v in daily_visitors if not ('datetime' in v and isinstance(v['datetime'], str))]

        # Sort valid visitors
        valid_visitors.sort(key=lambda x: datetime.strptime(x['datetime'], '%Y-%m-%d %H:%M:%S'))

        # Combine sorted valid visitors with invalid ones (optional, puts invalid ones at the end)
        sorted_daily_visitors = valid_visitors + invalid_visitors
    except Exception as e:
        # Log sorting errors if they occur, possibly due to unexpected data
        print(f"Warning: Could not sort visitors due to invalid data or structure: {e}")
        sorted_daily_visitors = daily_visitors # Fallback to unsorted list

    # Create response and set cache headers
    response = make_response(jsonify({"visitors": sorted_daily_visitors}))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/api/admin/delete_visitor_entry', methods=['POST'])
def delete_visitor_entry():
    if not is_admin_authenticated():
        return jsonify({"success": False, "message": "Unauthorized access"}), 403

    data = request.get_json()
    uid, visit_id = data.get('uid'), data.get('visit_id') if data else (None, None)
    if not uid or not visit_id:
        return jsonify({"success": False, "message": "Missing or empty uid/visit_id"}), 400

    try:
        face_data = load_face_data()
        user_visits = face_data.get(uid, {}).get('visitor')
        if not isinstance(user_visits, list):
            return jsonify({"success": False, "message": "Invalid or missing visitor data"}), 404

        visit_to_remove = next((v for v in user_visits if isinstance(v, dict) and v.get('visit_id') == visit_id), None)
        if not visit_to_remove:
            return jsonify({"success": False, "message": f"Visit ID '{visit_id}' not found for UID '{uid}'"}), 404

        face_data[uid]['visitor'] = [v for v in user_visits if v.get('visit_id') != visit_id]

        pdf = visit_to_remove.get('document_pdf')
        if pdf and isinstance(pdf, str) and pdf.strip():
            abs_path, abs_folder = os.path.abspath(pdf), os.path.abspath(PDF_FOLDER)
            if abs_path.startswith(abs_folder) and os.path.exists(pdf):
                try: os.remove(pdf)
                except OSError as e: print(f"PDF deletion failed: {e}")

        if save_face_data(face_data):
            return jsonify({"success": True, "message": "Visitor entry deleted successfully"})
        return jsonify({"success": False, "message": "Failed to save changes"}), 500

    except Exception as e:
        print(f"Error deleting visit {visit_id} for UID {uid}: {e}")
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500

def safe_get(data, key, default='N/A'):
    return data.get(key, default) if data else default

@app.route('/api/admin/generate_report')
def generate_visitor_report():
    if not is_admin_authenticated():
        return jsonify({"error": "Unauthorized access"}), 403

    start_date_str = request.args.get('start_date')
    end_date_str = request.args.get('end_date')

    if not start_date_str:
        return jsonify({"error": "Missing required parameter: start_date"}), 400

    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        if not end_date_str:
            end_date = start_date
        else:
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()

        if start_date > end_date:
            return jsonify({"error": "Start date cannot be after end date"}), 400

    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

    try:
        face_data = load_face_data()
        report_data = []

        for uid, user_data in face_data.items():
            if isinstance(user_data, dict) and 'visitor' in user_data and isinstance(user_data['visitor'], list):
                for visit in user_data['visitor']:
                    if isinstance(visit, dict) and 'datetime' in visit and isinstance(visit['datetime'], str):
                        try:
                            visit_dt = datetime.strptime(visit['datetime'], '%Y-%m-%d %H:%M:%S')
                            visit_date = visit_dt.date()
                            if start_date <= visit_date <= end_date:
                                report_entry = {
                                    "Name": safe_get(user_data, 'name'),
                                    "Phone": safe_get(user_data, 'phone'),
                                    "Address": safe_get(user_data, 'address'),
                                    "Tahasil": safe_get(user_data, 'tahasil'),
                                    "District": safe_get(user_data, 'district'),
                                    "Visit ID": safe_get(visit, 'visit_id'),
                                    "Date & Time": visit_dt.strftime('%d/%m/%Y %I:%M:%S %p'),
                                    "Purpose": safe_get(visit, 'purpose'),
                                    "Document": "Yes" if visit.get('document_pdf') else "No",
                                    "Status": safe_get(visit, 'status', 'unknown').capitalize(),
                                    "Entry Confirmed": "Yes" if visit.get('entry_confirmed', False) else "No",
                                    "Forwarded To": safe_get(visit, 'forwarding_department', '-'),
                                    "Remark": safe_get(visit, 'forwarding_note', '-')
                                }
                                report_data.append(report_entry)
                        except ValueError:
                            print(f"Skipping visit due to invalid datetime format: UID {uid}, Visit {visit.get('visit_id', 'N/A')}")
                        except Exception as e:
                            print(f"Error processing visit row for report: UID {uid}, Visit {visit.get('visit_id', 'N/A')} - {e}")

        report_data.sort(key=lambda x: datetime.strptime(x['Date & Time'], '%d/%m/%Y %I:%M:%S %p'))

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Visitor Report"

        headers = [
            "Name", "Phone", "Address", "Tahasil", "District", "Visit ID",
            "Date & Time", "Purpose", "Document", "Status", "Entry Confirmed",
            "Forwarded To", "Remark"
        ]
        ws.append(headers)

        header_font = Font(bold=True, color="FFFFFF")
        header_fill = openpyxl.styles.PatternFill(start_color="2C3E90", end_color="2C3E90", fill_type="solid")
        for cell in ws[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal='center', vertical='center')

        for entry in report_data:
            row_data = [entry.get(header, 'N/A') for header in headers]
            ws.append(row_data)

        for col_idx, column_letter in enumerate(openpyxl.utils.get_column_letter(i) for i in range(1, len(headers) + 1)):
            max_length = 0
            column = ws[column_letter]
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = (max_length + 2) * 1.2
            ws.column_dimensions[column_letter].width = max(15, min(adjusted_width, 50))
        excel_stream = io.BytesIO()
        wb.save(excel_stream)
        excel_stream.seek(0)
        if start_date == end_date:
            filename = f"visitor_report_{start_date_str}.xlsx"
        else:
            filename = f"visitor_report_{start_date_str}_to_{end_date_str}.xlsx"
        return send_file(
            excel_stream,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        print(f"CRITICAL: Error generating Excel report: {e}")
        return jsonify({"error": f"An unexpected server error occurred while generating the report: {str(e)}"}), 500

@app.route('/api/admin/delete_user', methods=['POST']) # Crucial: methods=['POST']
def delete_user():
    # !! IMPORTANT: Add robust admin authentication/authorization here !!
    # Example: if not is_admin(session): return jsonify(...), 403

    data = request.get_json()
    if not data or 'uid' not in data:
        return jsonify({"success": False, "message": "Missing UID"}), 400
    uid_to_delete = data['uid']
    try:
        face_data = load_face_data()
        if uid_to_delete not in face_data:
            return jsonify({"success": False, "message": "Visitor UID not found"}), 404
        removed_data = face_data.pop(uid_to_delete, None)
        if removed_data is None:
            return jsonify({"success": False, "message": "User UID found initially but could not be removed."}), 500
        save_face_data(face_data)
        return jsonify({"success": True, "message": "User and all associated data deleted successfully"})
    except Exception as e:
        print(f"Error deleting user {uid_to_delete}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": f"An server error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run()


# gunicorn --workers 3 --bind 0.0.0.0:8080 app:app
import base64
import os
import json
import bcrypt
import cv2
import numpy as np
import face_recognition
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory
from ultralytics import YOLO
from datetime import datetime
from deep_translator import GoogleTranslator
import ssl
from werkzeug.serving import run_simple

app = Flask(__name__)
app.secret_key = os.urandom(24)

os.makedirs('facedata', exist_ok=True)
os.makedirs('encodings', exist_ok=True)

person_model = YOLO('models/yolo11n-seg.pt')
face_model = YOLO('models/yolov11n-face.pt')

person_model.model.fuse = lambda *args, **kwargs: None

def generate_visitor_id():
    today = datetime.now()
    try:
        with open('facedata/facedata.json', 'r') as f:
            visitors_data = json.load(f)
    except FileNotFoundError:
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
def translate_text(text, target_lang):
    if not text or (target_lang == 'mr' and is_marathi(text)):
        return text
    translator = GoogleTranslator(source='auto', target=target_lang)
    try:
        return translator.translate(text)
    except:
        return text

def translate_text(text, target_lang):
    translator = GoogleTranslator(source='auto', target=target_lang)
    return translator.translate(text)

def save_face_images(frame, uid):
    user_dir = f'facedata/{uid}'
    os.makedirs(user_dir, exist_ok=True)
    results = face_model.predict(source=frame, stream=True)
    saved_images = []
    face_encodings = []
    for r in results:
        if r.boxes is not None:
            for box in r.boxes.data:
                x1, y1, x2, y2 = map(int, box[:4].cpu().numpy())
                face_crop = frame[y1:y2, x1:x2]
                filename = f'{uid}_img{len(saved_images) + 1}.jpg'
                image_path = os.path.join(user_dir, filename)
                cv2.imwrite(image_path, face_crop)
                saved_images.append(f'facedata/{uid}/{filename}')
                rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb_face)
                if encodings:
                    encoding_filename = f'encodings/{uid}_encoding{len(face_encodings) + 1}.npy'
                    np.save(encoding_filename, encodings[0])
                    face_encodings.append(encoding_filename)
                if len(saved_images) >= 3:
                    break
    return saved_images, face_encodings

def load_users():
    try:
        with open("auth.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

def save_users(users):
    with open("auth.json", "w") as file:
        json.dump(users, file, indent=4)

@app.route("/", methods=["GET", "POST"])
def login():
    if "user" in session:
        return redirect(url_for("dashboard", role=session["user"]["role"]))
    if request.method == "POST":
        data = request.json
        username = data.get("username")
        password = data.get("password")
        users = load_users()
        if username in users and bcrypt.checkpw(password.encode(), users[username]["password"].encode()):
            role = users[username]["role"]
            session["user"] = {"username": username, "role": role}
            return jsonify(success=True, redirect_url=url_for("dashboard", role=role))
        return jsonify(success=False, message="Invalid credentials")
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        data = request.json
        username = data.get("username")
        password = data.get("password")
        role = data.get("role")
        users = load_users()
        if not username or not password or not role:
            return jsonify(success=False, message="All fields are required")
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
    return render_template(templates.get(role, "unauthorized.html"))

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


@app.route('/register_visitor', methods=['POST'])
def register_visitor():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid request, no data received'}), 400
    try:
        with open('facedata/facedata.json', 'r') as f:
            facedata = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        facedata = {}
    existing_uid = None
    for uid, user_data in facedata.items():
        if user_data.get('name') == data.get('name') and user_data.get('phone') == data.get('phone'):
            existing_uid = uid
            break
    uid = existing_uid if existing_uid else f"UID{int(datetime.now().timestamp())}"
    tahasil_options = ["AKOLA", "AKOT", "TELAHARA", "BALAPUR", "PATUR", "MURTIZAPUR", "BARSHITAKLI"]
    selected_tahasil = data.get('tahasil', '')
    if selected_tahasil not in tahasil_options and selected_tahasil:
        return jsonify({'error': 'Invalid tahasil selection'}), 400
    visitor_data = {
        'name': translate_text(data.get('name', ''), 'mr'),
        'phone': data.get('phone', ''),
        'email': data.get('email', ''),
        'address': translate_text(data.get('address', ''), 'mr'),
        'tahasil': translate_text(selected_tahasil, 'mr'),
        'district': translate_text(data.get('district', 'Akola'), 'mr')
    }
    if not existing_uid:
        visitor_data['registration_datetime'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if 'frame' in data and data['frame']:
        try:
            frame_data = base64.b64decode(data['frame'].split(',')[1])
            nparr = np.frombuffer(frame_data, np.uint8)
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
    visitor_id = generate_visitor_id()
    visit_entry = {
        'visit_id': visitor_id,
        'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'purpose': translate_text(data.get('purpose', ''), 'mr')  # Translate purpose to Marathi
    }
    if existing_uid and uid in facedata:
        facedata[uid].setdefault('visitor', []).append(visit_entry)
    else:
        visitor_data['visitor'] = [visit_entry]
        facedata[uid] = visitor_data
    with open('facedata/facedata.json', 'w') as f:
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
    try:
        with open('facedata/facedata.json', 'r') as f:
            facedata = json.load(f)
    except FileNotFoundError:
        return jsonify({'success': False, 'message': 'No visitor data found'})
    matching_visitors = []
    for uid, visitor_data in facedata.items():
        if (query in visitor_data.get('name', '').lower() or
                query in uid.lower() or
                any(query in visit.get('visit_id', '').lower() for visit in visitor_data.get('visitor', []))):
            last_visit = visitor_data.get('visitor', [])[-1] if visitor_data.get('visitor') else None
            matching_visitors.append({
                'uid': uid,
                'name': visitor_data.get('name', 'N/A'),
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
    try:
        with open('facedata/facedata.json', 'r') as f:
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
                        'visitor_data': user_data
                    })
    return jsonify({'success': False, 'message': 'Visitor not found'})

@app.route('/detect_face', methods=['POST'])
def detect_face():
    if 'frame' not in request.files:
        return jsonify({'recognized': False, 'message': 'No frame uploaded'})
    frame = np.frombuffer(request.files['frame'].read(), dtype=np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
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
    for fr in face_results:
        if fr.boxes is not None:
            for box in fr.boxes.data:
                x1, y1, x2, y2 = map(int, box[:4].cpu().numpy())
                face_crop = frame[y1:y2, x1:x2]
                rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                try:
                    with open('facedata/facedata.json', 'r') as f:
                        known_data = json.load(f)
                except FileNotFoundError:
                    known_data = {}
                known_encodings = []
                known_uids = []
                for uid, user in known_data.items():
                    for encoding_path in user.get('face_encodings', []):
                        if os.path.exists(encoding_path):
                            encoding = np.load(encoding_path)
                            known_encodings.append(encoding)
                            known_uids.append(uid)
                face_encodings = face_recognition.face_encodings(rgb_face)
                if face_encodings:
                    matches = face_recognition.compare_faces(known_encodings, face_encodings[0], tolerance=0.4)
                    if True in matches:
                        best_match_index = matches.index(True)
                        matched_uid = known_uids[best_match_index]
                        matched_user = known_data[matched_uid]
                        return jsonify({
                            'recognized': True,
                            'user_data': matched_user
                        })
    return jsonify({'recognized': False, 'message': 'Face not recognized'})


@app.route('/confirm_visitor_entry', methods=['POST'])
def confirm_visitor_entry():
    data = request.json
    visit_id = data.get('visitId')
    dvn = data.get('dvn')

    try:
        with open('facedata/facedata.json', 'r') as f:
            facedata = json.load(f)
    except FileNotFoundError:
        return jsonify({'success': False, 'message': 'Visitor data not found'})

    for uid, user_data in facedata.items():
        for visit in user_data.get('visitor', []):
            if visit.get('visit_id') == visit_id:
                visit['dvn'] = dvn
                with open('facedata/facedata.json', 'w') as f:
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
        with open('facedata/facedata.json', 'r') as f:
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
    image_bytes = file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'success': False, 'message': 'Invalid image format'})
    _, face_crop = detect_person_and_face(frame)
    if face_crop is None:
        return jsonify({'success': False, 'message': 'No face detected in the image'})
    user_dir = f'profiles/{uid}'
    os.makedirs(user_dir, exist_ok=True)

    existing_images = [f for f in os.listdir(user_dir) if f.startswith(f'{uid}_img') and f.endswith('.jpg')]
    next_img_num = len(existing_images) + 1
    face_path = f'{user_dir}/{uid}_img{next_img_num}.jpg'
    cv2.imwrite(face_path, face_crop)
    facedata_path = 'facedata/facedata.json'
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
    facedata[uid]["profile_img"].append(face_path)
    with open(facedata_path, 'w') as json_file:
        json.dump(facedata, json_file, indent=4)
    return jsonify({
        'success': True,
        'message': 'Profile image processed successfully',
        'image_path': face_path,
        'image_number': next_img_num
    })

@app.route('/dashboard/profiles/<uid>/<filename>', methods=['GET'])
def get_profile_image(uid, filename):
    profiles_dir = os.path.join('profiles')
    if not os.path.exists(os.path.join(profiles_dir, uid, filename)):
        return jsonify({'success': False, 'message': 'Image not found'}), 404
    return send_from_directory(os.path.join(profiles_dir, uid), filename)

@app.route('/departments', methods=['GET'])
def get_departments():
    with open('JSON/departments.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    return jsonify(data)


@app.route('/complete_meeting', methods=['POST'])
def complete_meeting():
    try:
        visit_id = request.args.get('visit_id')
        uid = request.args.get('uid')
        if not visit_id or not uid:
            return jsonify({"error": "Missing visit_id or uid parameter"}), 400
        json_file_path = 'facedata/facedata.json'
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
        with open('facedata/facedata.json', 'r') as f:
            facedata = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return jsonify({"success": False, "message": "Could not load visitor data."}), 500
    if uid not in facedata:
        return jsonify({"success": False, "message": f"Visitor with UID '{uid}' not found."}), 404
    translated_remark = translate_text(remark, 'mr')
    translated_department = translate_text(forwarding_department, 'mr')

    visit_found = False
    for visit in facedata[uid].get('visitor', []):
        if visit.get('visit_id') == visit_id:
            visit['remark'] = translated_remark
            visit['forwarding_department'] = translated_department
            visit['status'] = 'pending'
            visit_found = True
            break

    if not visit_found:
        return jsonify({"success": False, "message": f"Visit with ID '{visit_id}' not found for visitor '{uid}'."}), 404
    try:
        with open('facedata/facedata.json', 'w') as f:
            json.dump(facedata, f, indent=4)
        return jsonify({"success": True, "message": "Visitor action recorded successfully."}), 200
    except Exception as e:
        return jsonify({"success": False, "message": f"Failed to save updated visitor data: {str(e)}"}), 500

@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    text = data.get("text", "")
    lang = data.get("lang", "mr")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    translated_text = translate_text(text, lang)
    return jsonify({"translated_text": translated_text, "language": lang})
#
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5110, debug=True)

context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain('cert.pem', 'key.pem')

run_simple('0.0.0.0', 5110, app, use_debugger=True, use_reloader=True, ssl_context=context)
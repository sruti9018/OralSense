from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import base64
import os
import database

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_MODEL_PATH = r"C:\Users\Sruti Sivashankar\OneDrive\Desktop\oral_cancer_ai\models\best_model.h5"
MULTI_MODEL_PATH = r"C:\Users\Sruti Sivashankar\OneDrive\Desktop\oral_cancer_ai\models\multimodal_model.h5"

print("Loading models...")
image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH, compile=False)
image_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
multi_model = tf.keras.models.load_model(MULTI_MODEL_PATH, compile=False)
multi_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Models loaded!")

# Class order from retrained augmented model: Normal=0, OC=1, OPMD=2, Variations=3
CLASS_NAMES = ['Normal', 'OC', 'OPMD', 'Variations']
CLASS_INFO = {
    'Normal':     {'color': '#22c55e', 'risk': 'Low',     'risk_order': 0, 'desc': 'No signs of malignancy detected.',                      'risk_key': 'normal'},
    'Variations': {'color': '#f59e0b', 'risk': 'Low-Med', 'risk_order': 1, 'desc': 'Minor variations from normal tissue.',                  'risk_key': 'low'},
    'OPMD':       {'color': '#f97316', 'risk': 'Medium',  'risk_order': 2, 'desc': 'Oral Potentially Malignant Disorder detected.',         'risk_key': 'medium'},
    'OC':         {'color': '#ef4444', 'risk': 'High',    'risk_order': 3, 'desc': 'Oral Cancer indicators detected. Seek immediate care.', 'risk_key': 'high'}
}

RISK_ORDER = {'normal': 0, 'low': 1, 'medium': 2, 'high': 3}
IMG_SIZE   = 224
MC_RUNS    = 20

def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img   = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img   = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def mc_predict_image(img_array, n_runs=MC_RUNS):
    preds = np.array([image_model(img_array, training=True).numpy() for _ in range(n_runs)])
    return np.mean(preds, axis=0)[0], np.std(preds, axis=0)[0]

def mc_predict_multi(img_array, meta, n_runs=MC_RUNS):
    preds = np.array([
        multi_model({'image_input': img_array, 'meta_input': meta}, training=True).numpy()
        for _ in range(n_runs)
    ])
    return np.mean(preds, axis=0)[0], np.std(preds, axis=0)[0]

def uncertainty_label(std):
    if std < 2.0:    return "Very Low"
    elif std < 5.0:  return "Low"
    elif std < 10.0: return "Moderate"
    else:            return "High"

def get_best_class(mean_pred):
    """Always return the highest risk class above threshold."""
    # Check each class from highest to lowest risk
    # OC=1, OPMD=2, Variations=3, Normal=0
    if mean_pred[1] >= 0.03:  # OC >= 3%
        return 1
    if mean_pred[2] >= 0.05:  # OPMD >= 5%
        return 2
    if mean_pred[3] >= 0.05:  # Variations >= 5%
        return 3
    return int(np.argmax(mean_pred))

def get_final_class(img_pred, multi_pred):
    """
    Always use the higher risk result between image and multimodal.
    In cancer screening, never downgrade — always take the more cautious result.
    """
    img_cls   = get_best_class(img_pred)
    multi_cls = get_best_class(multi_pred)
    img_risk   = CLASS_INFO[CLASS_NAMES[img_cls]]['risk_order']
    multi_risk = CLASS_INFO[CLASS_NAMES[multi_cls]]['risk_order']
    # Return whichever model detected higher risk
    if img_risk >= multi_risk:
        return img_cls, img_pred
    else:
        return multi_cls, multi_pred

def get_gradcam(img_array):
    try:
        last_conv = None
        for layer in reversed(image_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer.name
                break
        grad_model = tf.keras.models.Model(
            inputs=image_model.input,
            outputs=[image_model.get_layer(last_conv).output, image_model.output]
        )
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_array)
            pred_idx        = tf.argmax(preds[0])
            class_channel   = preds[:, pred_idx]
        grads       = tape.gradient(class_channel, conv_out)
        pooled      = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap     = conv_out[0] @ pooled[..., tf.newaxis]
        heatmap     = tf.squeeze(heatmap)
        heatmap     = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap_np  = heatmap.numpy()
        heatmap_np  = cv2.resize(heatmap_np, (IMG_SIZE, IMG_SIZE))
        heatmap_col = cv2.applyColorMap(np.uint8(255 * heatmap_np), cv2.COLORMAP_JET)
        orig        = (img_array[0] * 255).astype(np.uint8)
        orig_bgr    = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
        overlay     = cv2.addWeighted(orig_bgr, 0.6, heatmap_col, 0.4, 0)
        _, buffer   = cv2.imencode('.jpg', overlay)
        return base64.b64encode(buffer).decode('utf-8')
    except:
        return None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file      = request.files['image']
        img_bytes = file.read()
        img_array = preprocess_image(img_bytes)

        age      = float(request.form.get('age',      40))
        sex      = float(request.form.get('sex',      0.5))
        smoking  = float(request.form.get('smoking',  0.0))
        chewing  = float(request.form.get('chewing',  0.0))
        arecanut = float(request.form.get('arecanut', 0.0))
        alcohol  = float(request.form.get('alcohol',  0.0))

        age_norm = min(max((age - 10) / 80, 0), 1)
        meta     = np.array([[age_norm, sex, smoking, chewing, arecanut, alcohol]], dtype=np.float32)

        img_mean,   img_std   = mc_predict_image(img_array)
        multi_mean, multi_std = mc_predict_multi(img_array, meta)

        # Individual model results
        img_cls  = get_best_class(img_mean)
        img_conf = float(img_mean[img_cls])  * 100
        img_unc  = float(img_std[img_cls])   * 100

        multi_cls  = get_best_class(multi_mean)
        multi_conf = float(multi_mean[multi_cls]) * 100
        multi_unc  = float(multi_std[multi_cls])  * 100

        # Final result = higher risk of the two
        final_cls, final_pred = get_final_class(img_mean, multi_mean)
        final_conf = float(final_pred[final_cls]) * 100
        final_std  = img_std if CLASS_INFO[CLASS_NAMES[img_cls]]['risk_order'] >= CLASS_INFO[CLASS_NAMES[multi_cls]]['risk_order'] else multi_std
        final_unc  = float(final_std[final_cls]) * 100

        gradcam_b64 = get_gradcam(img_array)

        return jsonify({
            'image_only': {
                'class':             CLASS_NAMES[img_cls],
                'confidence':        round(img_conf, 1),
                'uncertainty':       round(img_unc, 1),
                'uncertainty_label': uncertainty_label(img_unc),
                'color':             CLASS_INFO[CLASS_NAMES[img_cls]]['color'],
                'risk':              CLASS_INFO[CLASS_NAMES[img_cls]]['risk'],
                'risk_key':          CLASS_INFO[CLASS_NAMES[img_cls]]['risk_key'],
                'desc':              CLASS_INFO[CLASS_NAMES[img_cls]]['desc'],
                'probs':             {CLASS_NAMES[i]: round(float(img_mean[i])*100, 1) for i in range(4)}
            },
            'multimodal': {
                'class':             CLASS_NAMES[final_cls],
                'confidence':        round(final_conf, 1),
                'uncertainty':       round(final_unc, 1),
                'uncertainty_label': uncertainty_label(final_unc),
                'color':             CLASS_INFO[CLASS_NAMES[final_cls]]['color'],
                'risk':              CLASS_INFO[CLASS_NAMES[final_cls]]['risk'],
                'risk_key':          CLASS_INFO[CLASS_NAMES[final_cls]]['risk_key'],
                'desc':              CLASS_INFO[CLASS_NAMES[final_cls]]['desc'],
                'probs':             {CLASS_NAMES[i]: round(float(multi_mean[i])*100, 1) for i in range(4)}
            },
            'gradcam': gradcam_b64,
            'mc_runs': MC_RUNS
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save_scan', methods=['POST'])
def save_scan():
    try:
        data   = request.json
        conn   = database.get_db()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO patients (patient_id, name, age, sex, phone, address)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(patient_id) DO UPDATE SET
                name=excluded.name, age=excluded.age, sex=excluded.sex,
                phone=excluded.phone, address=excluded.address
        ''', (
            data['patient_id'], data.get('name'), data.get('age'),
            data.get('sex'), data.get('phone'), data.get('address')
        ))

        cursor.execute('''
            SELECT risk_key FROM scans WHERE patient_id=? ORDER BY created_at DESC LIMIT 1
        ''', (data['patient_id'],))
        last = cursor.fetchone()
        risk_increased = False
        if last:
            prev_risk = RISK_ORDER.get(last['risk_key'], 0)
            curr_risk = RISK_ORDER.get(data['multimodal']['risk_key'], 0)
            risk_increased = curr_risk > prev_risk

        cursor.execute('''
            INSERT INTO scans (
                patient_id, scan_date, doctor, department, clinical_notes,
                smoking, chewing, arecanut, alcohol,
                img_class, img_confidence, img_uncertainty, img_uncertainty_label,
                multi_class, multi_confidence, multi_uncertainty, multi_uncertainty_label,
                risk_key, gradcam
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ''', (
            data['patient_id'], data.get('scan_date'), data.get('doctor'),
            data.get('department'), data.get('clinical_notes'),
            int(data.get('smoking', 0)), int(data.get('chewing', 0)),
            int(data.get('arecanut', 0)), int(data.get('alcohol', 0)),
            data['image_only']['class'], data['image_only']['confidence'],
            data['image_only']['uncertainty'], data['image_only']['uncertainty_label'],
            data['multimodal']['class'], data['multimodal']['confidence'],
            data['multimodal']['uncertainty'], data['multimodal']['uncertainty_label'],
            data['multimodal']['risk_key'], data.get('gradcam')
        ))

        conn.commit()
        conn.close()
        return jsonify({'success': True, 'risk_increased': risk_increased})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/patient_history/<patient_id>', methods=['GET'])
def patient_history(patient_id):
    try:
        conn   = database.get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM patients WHERE patient_id=?', (patient_id,))
        patient = cursor.fetchone()
        cursor.execute('SELECT * FROM scans WHERE patient_id=? ORDER BY created_at ASC', (patient_id,))
        scans = cursor.fetchall()
        conn.close()
        if not patient:
            return jsonify({'error': 'Patient not found'}), 404
        scans_list = []
        for i, s in enumerate(scans):
            s_dict = dict(s)
            if i > 0:
                prev_risk = RISK_ORDER.get(scans[i-1]['risk_key'], 0)
                curr_risk = RISK_ORDER.get(s['risk_key'], 0)
                s_dict['risk_increased'] = curr_risk > prev_risk
                s_dict['risk_decreased'] = curr_risk < prev_risk
            else:
                s_dict['risk_increased'] = False
                s_dict['risk_decreased'] = False
            scans_list.append(s_dict)
        return jsonify({'patient': dict(patient), 'scans': scans_list, 'total_scans': len(scans_list)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/patients', methods=['GET'])
def list_patients():
    try:
        conn   = database.get_db()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT p.*, COUNT(s.id) as scan_count, MAX(s.created_at) as last_scan,
                   (SELECT risk_key FROM scans WHERE patient_id=p.patient_id ORDER BY created_at DESC LIMIT 1) as latest_risk
            FROM patients p
            LEFT JOIN scans s ON s.patient_id = p.patient_id
            GROUP BY p.patient_id ORDER BY last_scan DESC
        ''')
        patients = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return jsonify({'patients': patients})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/history')
def history_page():
    return send_from_directory(BASE_DIR, 'patient_history.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=False, port=5000)
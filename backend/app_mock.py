from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
from werkzeug.utils import secure_filename
import json
from PIL import Image
import numpy as np
import os.path as osp
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
except Exception:
    TrOCRProcessor = None
    VisionEncoderDecoderModel = None
try:
    from sentence_transformers import SentenceTransformer, util as st_util
except Exception:
    SentenceTransformer = None
    st_util = None
try:
    import sympy as sp
except Exception:
    sp = None

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def serve_root():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(base_dir, 'test.html')

@app.route('/test.html')
def serve_test_html():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(base_dir, 'test.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_upload_path(filename):
    return os.path.join(app.config['UPLOAD_FOLDER'], filename)

class LocalOCREngine:
    def __init__(self):
        self.model = None
        self.processor = None
        self._load_local_model()

    def _load_local_model(self):
        try:
            local_dir = 'models/trocr_math'
            if VisionEncoderDecoderModel is None or TrOCRProcessor is None:
                print('Transformers not available; OCR disabled')
                return
            if os.path.exists(local_dir) and os.path.isdir(local_dir):
                print('Loading local fine-tuned TrOCR from models/trocr_math')
                self.processor = TrOCRProcessor.from_pretrained(local_dir)
                self.model = VisionEncoderDecoderModel.from_pretrained(local_dir)
            else:
                print('Local model not found at models/trocr_math; falling back to base handwritten model')
                self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
                self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
        except Exception as e:
            print(f'Failed to load OCR model: {e}')
            self.processor = None
            self.model = None

    def _load_image(self, path):
        try:
            if path.lower().endswith('.pdf'):
                from pdf2image import convert_from_path
                images = convert_from_path(path, dpi=300)
                image = images[0]
                return image.convert('RGB')
            else:
                img = Image.open(path).convert('RGB')
                return img
        except Exception as e:
            print(f'OCR image load error for {path}: {e}')
            return None

    def extract_text(self, filepath):
        if not self.model or not self.processor:
            return ''

# -------------------------
# Hybrid grading components
# -------------------------

def split_sentences(text):
    parts = [p.strip() for p in text.replace("\n", " ").split('.')]
    return [p for p in parts if len(p) > 0]

def extract_formulas(text):
    import re
    lines = [l.strip() for l in text.split('\n')]
    formula_like = []
    pattern = re.compile(r"[=+\-*/^]|")
    # Simple heuristic: lines with math operators or variable assignments
    for l in lines:
        if any(op in l for op in ['=', '+', '-', '*', '/', '^']) and any(ch.isalpha() for ch in l):
            formula_like.append(l)
    # Also sweep inline expressions like y = 0.613x + 13.87
    inline = re.findall(r"[A-Za-z0-9_\s]*=\s*[^,;]+", text)
    formula_like.extend([i.strip() for i in inline])
    # Deduplicate while keeping order
    seen = set()
    ordered = []
    for f in formula_like:
        if f not in seen:
            seen.add(f)
            ordered.append(f)
    return ordered

class SemanticScorer:
    def __init__(self):
        self.model = None
        if SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            except Exception as e:
                print(f"SentenceTransformer load failed: {e}")
                self.model = None

    def score(self, student_sentences, key_sentences):
        if not student_sentences or not key_sentences:
            return 0.0
        if not self.model or st_util is None:
            # Fallback: Jaccard on token sets averaged
            def jacc(a, b):
                sa, sb = set(a.lower().split()), set(b.lower().split())
                if not sa or not sb:
                    return 0.0
                return len(sa & sb) / len(sa | sb)
            sims = [max(jacc(s, k) for k in key_sentences) for s in student_sentences]
            return float(sum(sims) / len(sims))
        try:
            emb_s = self.model.encode(student_sentences, convert_to_tensor=True, normalize_embeddings=True)
            emb_k = self.model.encode(key_sentences, convert_to_tensor=True, normalize_embeddings=True)
            cos = st_util.cos_sim(emb_s, emb_k)  # [S,K]
            # For each student sentence, take best matching key sentence
            best_per_s = cos.max(dim=1).values.cpu().numpy().tolist()
            return float(sum(best_per_s) / len(best_per_s))
        except Exception as e:
            print(f"Semantic scoring failed: {e}")
            return 0.0

class SymbolicScorer:
    def __init__(self):
        self.enabled = sp is not None

    def _normalize(self, expr_str):
        if not self.enabled:
            return None
        try:
            # Replace caret with ** for exponentiation
            expr_str = expr_str.replace('^', '**')
            # Sympy parsing with implicit multiplication
            return sp.sympify(expr_str, evaluate=False)
        except Exception:
            return None

    def equivalence(self, stu, key):
        if not self.enabled:
            return 0.0
        e1 = self._normalize(stu)
        e2 = self._normalize(key)
        if e1 is None or e2 is None:
            return 0.0
        try:
            diff = sp.simplify(e1 - e2)
            return 1.0 if diff == 0 else 0.0
        except Exception:
            try:
                return 1.0 if sp.simplify(sp.Eq(e1, e2)) is sp.true else 0.0
            except Exception:
                return 0.0

    def score(self, stu_formulas, key_formulas):
        if not stu_formulas or not key_formulas:
            return 0.0
        scores = []
        for sf in stu_formulas:
            best = 0.0
            for kf in key_formulas:
                best = max(best, self.equivalence(sf, kf))
                if best == 1.0:
                    break
            scores.append(best)
        return float(sum(scores) / len(scores))

semantic_scorer = SemanticScorer()
symbolic_scorer = SymbolicScorer()

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'OCR Exam Grading System API is running!',
        'mode': 'mock_ocr_ready'
    })

@app.route('/api/upload-question', methods=['POST'])
def upload_question():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if file and allowed_file(file.filename):
            file_id = str(uuid.uuid4())
            filename = secure_filename(f"{file_id}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return jsonify({
                'success': True,
                'file_id': file_id,
                'filename': filename,
                'paper_type': 'question',
                'message': 'Question paper uploaded successfully'
            })
        return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/upload-answer-key', methods=['POST'])
def upload_answer_key():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        answer_key_text = request.form.get('answer_key_text', '')
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if file and allowed_file(file.filename):
            file_id = str(uuid.uuid4())
            filename = secure_filename(f"{file_id}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Persist provided answer key text if any
            if answer_key_text:
                ak_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{file_id}_answer_key.txt")
                with open(ak_path, 'w', encoding='utf-8') as f:
                    f.write(answer_key_text)
            return jsonify({
                'success': True,
                'file_id': file_id,
                'filename': filename,
                'paper_type': 'answer_key',
                'has_text': bool(answer_key_text),
                'message': 'Model answer key uploaded successfully'
            })
        return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        paper_type = request.form.get('paper_type', 'answer')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            file_id = str(uuid.uuid4())
            filename = secure_filename(f"{file_id}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            file.save(filepath)
            
            return jsonify({
                'success': True,
                'file_id': file_id,
                'filename': filename,
                'paper_type': paper_type,
                'message': 'File uploaded successfully'
            })
        
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/files/<path:filename>')
def serve_uploaded(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

ocr_engine = LocalOCREngine()

@app.route('/api/process-paper', methods=['POST'])
def process_paper():
    try:
        data = request.get_json()
        filename = data.get('filename')
        paper_type = data.get('paper_type', 'answer')
        
        # Real OCR via local engine (falls back to empty string if unavailable)
        filepath = get_upload_path(filename) if filename else None
        extracted_text = ocr_engine.extract_text(filepath) if filepath and os.path.exists(filepath) else ''
        
        # Simple dynamic placeholder grading based on extracted text length
        word_count = len(extracted_text.split()) if extracted_text else 0
        coverage = min(1.0, word_count / 250.0)  # assume ~250 words ~ full coverage
        percentage = round(coverage * 100, 2)
        total_score = round((percentage / 100) * 20, 1)
        grading_results = {
            'total_score': total_score,
            'max_total_score': 20,
            'percentage': percentage,
            'question_grades': {
                'Overall': {
                    'score': total_score,
                    'max_score': 20,
                    'percentage': percentage,
                    'feedback': [
                        f'📝 Detected word count: {word_count}',
                        'ℹ️ For detailed correctness, use "Grade Against Model Answer Key"'
                    ]
                }
            },
            'summary': {
                'total_questions': 1,
                'correct_questions': 1 if percentage >= 70 else 0,
                'accuracy': percentage,
                'strengths': ['Good amount of extracted text' if word_count > 100 else 'OCR text limited'],
                'improvements': ['Upload clearer images or PDF for better OCR']
            },
            'pages': data.get('student_filenames') or ([filename] if filename else [])
        }
        
        response = {
            'success': True,
            'file_id': data.get('file_id'),
            'paper_type': paper_type,
            'processing_results': {
                'page_analysis': {
                    'total_regions': 4,
                    'region_types': ['question', 'answer', 'answer', 'answer']
                },
                'ocr_results': {
                    'regions': [{
                        'type': 'answer',
                        'text': extracted_text,
                        'confidence': 0.85,
                        'bbox': [50, 100, 750, 900]
                    }],
                    'full_text': extracted_text
                },
                'grading_results': grading_results if paper_type == 'answer' else None
            }
        }
        
        # Save results
        results_file = os.path.join(
            app.config['PROCESSED_FOLDER'], 
            f"{data.get('file_id')}_results.json"
        )
        
        with open(results_file, 'w') as f:
            json.dump(response, f, indent=2, default=str)
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/api/grade', methods=['POST'])
def grade_against_key():
    try:
        data = request.get_json()
        student_filename = data.get('student_filename')
        student_file_id = data.get('student_file_id')
        answer_key_filename = data.get('answer_key_filename')
        answer_key_file_id = data.get('answer_key_file_id')
        answer_key_text = data.get('answer_key_text', '')
        # New: arrays for multi-page images
        student_filenames = data.get('student_filenames') or []
        answer_key_filenames = data.get('answer_key_filenames') or []

        # OCR both student and answer key files using local engine (unless text is provided)
        def ocr_concat(filenames):
            texts = []
            for idx, fn in enumerate(filenames):
                p = get_upload_path(fn)
                if os.path.exists(p):
                    txt = ocr_engine.extract_text(p) or ''
                    texts.append(f"\n\n[Page {idx+1}]\n" + txt)
            return "".join(texts)

        if student_filenames:
            student_text = ocr_concat(student_filenames)
        else:
            student_path = get_upload_path(student_filename) if student_filename else None
            student_text = ocr_engine.extract_text(student_path) if student_path and os.path.exists(student_path) else ''

        # If answer_key_text not provided, try to read it from processed folder
        if not answer_key_text and answer_key_file_id:
            ak_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{answer_key_file_id}_answer_key.txt")
            if os.path.exists(ak_path):
                with open(ak_path, 'r', encoding='utf-8') as f:
                    answer_key_text = f.read()
        # If still not provided, OCR the uploaded answer key image/pdf
        if not answer_key_text:
            if answer_key_filenames:
                answer_key_text = ocr_concat(answer_key_filenames)
            elif answer_key_filename:
                ak_path_file = get_upload_path(answer_key_filename)
                if os.path.exists(ak_path_file):
                    answer_key_text = ocr_engine.extract_text(ak_path_file) or ''

        # Fallback mock model key if still empty
        if not answer_key_text:
            answer_key_text = "Regression equations, correlation r=0.863, correct hypothesis conclusion, and quality control improvement."

        def normalize(text):
            return ''.join(ch.lower() if ch.isalnum() or ch.isspace() else ' ' for ch in text)

        def jaccard(a, b):
            sa = set(normalize(a).split())
            sb = set(normalize(b).split())
            if not sa or not sb:
                return 0.0
            return len(sa & sb) / len(sa | sb)

        # Hybrid scoring
        stu_sent = split_sentences(student_text)
        key_sent = split_sentences(answer_key_text)
        sem_score = semantic_scorer.score(stu_sent, key_sent)
        stu_form = extract_formulas(student_text)
        key_form = extract_formulas(answer_key_text)
        sym_score = symbolic_scorer.score(stu_form, key_form)
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(None, normalize(student_text), normalize(answer_key_text)).ratio()
        jac = jaccard(student_text, answer_key_text)
        lex_score = 0.6 * ratio + 0.4 * jac
        hybrid = 0.5 * sym_score + 0.35 * sem_score + 0.15 * lex_score

        percentage = round(100 * hybrid, 2)
        total_score = round((percentage / 100) * 20, 1)

        grading_results = {
            'total_score': total_score,
            'max_total_score': 20,
            'percentage': percentage,
            'question_grades': {
                'Overall': {
                    'score': total_score,
                    'max_score': 20,
                    'percentage': percentage,
                    'feedback': [
                        f'🧮 Symbolic match: {round(sym_score*100,1)}%',
                        f'🧠 Semantic match: {round(sem_score*100,1)}%',
                        f'🔤 Lexical match: {round(lex_score*100,1)}%'
                    ]
                }
            },
            'summary': {
                'total_questions': 1,
                'correct_questions': 1 if percentage >= 70 else 0,
                'accuracy': percentage,
                'strengths': ['Content overlap with key terms' if percentage >= 60 else ''],
                'improvements': ['Align formulas and explanations with model answer']
            }
        }

        return jsonify({
            'success': True,
            'grading_results': grading_results,
            'student': {
                'filename': student_filename,
                'file_id': student_file_id
            },
            'answer_key': {
                'filename': answer_key_filename,
                'file_id': answer_key_file_id,
                'text_used': answer_key_text[:500]
            }
        })
    except Exception as e:
        return jsonify({'error': f'Grading failed: {str(e)}'}), 500

@app.route('/api/analyze-page', methods=['POST'])
def analyze_page():
    try:
        data = request.get_json()
        filename = data.get('filename')
        answer_key_text = data.get('answer_key_text', '')
        if not filename:
            return jsonify({'error': 'filename required'}), 400
        path = get_upload_path(filename)
        if not os.path.exists(path):
            return jsonify({'error': 'file not found'}), 404

        # OCR the page
        text = ocr_engine.extract_text(path) or ''

        # Split / extract
        sentences = split_sentences(text)
        formulas = extract_formulas(text)

        # Compare to key if provided
        analysis = {
            'text': text,
            'sentences': sentences,
            'formulas': formulas,
            'semantic_match': None,
            'symbolic_hits': [],
            'explanation': '',
            'issues': [],
            'suggestions': []
        }

        if answer_key_text:
            key_sent = split_sentences(answer_key_text)
            sem = semantic_scorer.score(sentences, key_sent)
            analysis['semantic_match'] = round(sem * 100, 2)
            key_forms = extract_formulas(answer_key_text)
            if key_forms:
                for f in formulas:
                    ok = max(symbolic_scorer.equivalence(f, kf) for kf in key_forms)
                    analysis['symbolic_hits'].append({'formula': f, 'match': bool(ok)})
                # Identify unmatched key formulas for feedback
                unmatched = []
                for kf in key_forms:
                    hit = any(symbolic_scorer.equivalence(kf, sf) for sf in formulas)
                    if not hit:
                        unmatched.append(kf)
                if unmatched:
                    analysis['issues'].append(f"Missing/incorrect formulas: {', '.join(unmatched[:5])}{'…' if len(unmatched)>5 else ''}")
                if sem < 0.6:
                    analysis['issues'].append('Explanatory text differs from model answer')
                if not formulas:
                    analysis['issues'].append('No formula detected on this page')
                # Suggestions
                if unmatched:
                    analysis['suggestions'].append('Show the full derivation and include the key equations explicitly')
                if sem < 0.6:
                    analysis['suggestions'].append('Align your reasoning steps with the standard method used in the key')
                if not formulas:
                    analysis['suggestions'].append('Write formulas clearly with = and standard symbols so OCR can read them')

        # Heuristic explanation from OCR text
        important_terms = []
        lower = text.lower()
        for term in ['regression','correlation','mean','variance','standard deviation','probability','binomial','poisson','normal','hypothesis','z','t','p-value','confidence','estimation']:
            if term in lower and term not in important_terms:
                important_terms.append(term)
        summary_sent = sentences[:4]
        explanation_parts = []
        if summary_sent:
            explanation_parts.append('Summary: ' + ' '.join(summary_sent))
        if important_terms:
            explanation_parts.append('Detected concepts: ' + ', '.join(important_terms))
        if formulas:
            preview = '; '.join(formulas[:3])
            explanation_parts.append('Key formulas detected: ' + preview + ('…' if len(formulas)>3 else ''))
        analysis['explanation'] = ' \n'.join(explanation_parts) if explanation_parts else 'Text is sparse; add clearer steps and formulas.'

        return jsonify({'success': True, 'analysis': analysis})
    except Exception as e:
        return jsonify({'error': f'analysis failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("🚀 Starting OCR Exam Grading System API (Mock Mode)")
    print("📄 Upload endpoint: POST /api/upload") 
    print("🔍 Process paper: POST /api/process-paper")
    print("💚 Health check: GET /api/health")
    print("🎭 Using mock OCR results for demonstration")
    print("🔗 Access at: http://localhost:5000")
    app.run(debug=True, port=5000, host='0.0.0.0')

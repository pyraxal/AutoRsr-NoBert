#endpoints map:
#
#transcription step -> takes a mp3 
#returns json of transcription.

#alignment step -> takes ground truth and output, if not provided assume default gt
#returning the aligned response and response. Will return "" if nothing is matchable/too wrong
#Do note, this matches SENTENCES. I have a option that will return best substring, so lmk ig
#See formatting later below

#edit-sequence/errors step -> takes aligned output in a certain format (shown below)
#scores" the output, returning BOTH the scores, and the errors made
#Ideally the frontend should have a visual that can take the errors made and 
#"unscramble" it to show the corrected sentence in a dropdown or something 
#errors, edit sequence, score in list form 

#scoring-step -> takes errors made, percentile and age and returns score and pass/fail

#master endpoint -> will take mp3 + text file + percentile and return the output of 
#transcription + alignment + scoring. This can be the "main" pipeline 
#and "playground" where you can test individual components by calling above.
#It will return a mega json, which I will attach in the repo as an example

#formatting
#transcription output will be formatted like this:
'''
{
        "Transcription": "the black ball washed onto the blue ocean. the quick fox brown jumps. hello world".
    }
'''

#aligned input:
'''
{
        "Transcription": "the black ball washed onto the blue ocean. the quick fox brown jumps. hello world."
        "Ground Truth":"The black ball washed into the blue ocean. The quick brown fox jumps. Hello World."
    }
'''
#aligned output:
'''
{
        "Sentences": [
            {
                "id": "1",
                "Ground Truth": "blue ball washed onto the black ocean",
                "Aligned": "the black ball washed onto the blue ocean. the quick fox brown jumps. hello world"
            },
            {
                "id": "2",
                "Ground Truth": "the quick brown fox jumps",
                "Aligned": "the quick fox brown jumps"
            },
            {
                "id": "3",
                "Ground Truth": "hello world",
                "Aligned": "hello wrold"
            }
        ]
'''

#scoring step like this:
'''
data = {
        "Sentences": [
            {
                "id": "1",
                "Sentence": "the black ball washed onto the blue ocean",
                "Errors": 2,
                "Score": 1,
                "Edit Script": {'Insertions': [], 'Deletions': [(0, 'the')], 'Substitutions': [], 'Swaps': [(0, 5, 'black', 'blue')]}
            },
            {
                "id": "2",
                "Sentence": "the quick brown fox jumps",
                "Errors": 1,
                "Score": 1,
                "Edit Script": {'Insertions': [], 'Deletions': [], 'Substitutions': [], 'Swaps': [(2, 3, 'fox', 'brown')]}
            },
            {
                "id": "3",
                "Sentence": "hello world",
                "Errors": 1,
                "Score": 1,
                "Edit Script": {'Insertions': [], 'Deletions': [], 'Substitutions': [], 'Swaps': [(2, 3, 'fox', 'brown')]}
            }
        ]
'''
from flask import Flask, request, jsonify
import os
import json
import re
from auto_rsr import transcribe_to_json, align_transcription_to_ground_truth, generate_edit_sequences_and_score, evaluate_rsr_result, run_full_rsr_analysis

app = Flask(__name__)

UPLOAD_FOLDER_WAV = 'uploads/wav_file'
UPLOAD_FOLDER_TEXT = 'uploads/text_file'
os.makedirs(UPLOAD_FOLDER_WAV, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_TEXT, exist_ok=True)

# --- TRANSCRIPTION ---
@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'wav_file' not in request.files:
        return jsonify({"error": "Missing WAV file"}), 400

    wav_file = request.files['wav_file']
    if wav_file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    wav_path = os.path.join(UPLOAD_FOLDER_WAV, wav_file.filename)
    wav_file.save(wav_path)

    try:
        result = transcribe_to_json(wav_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Route 2: Alignment Step ---
@app.route('/align', methods=['POST'])
def align():
    data = request.get_json()
    if not data or "Transcription" not in data:
        return jsonify({"error": "Missing 'Transcription' field in JSON"}), 400

    ground_truth = data.get("Ground Truth")  # Optional
    try:
        result = align_transcription_to_ground_truth(data, ground_truth_text=ground_truth)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Route 3: Edit Sequence / Scoring Step ---
@app.route('/edit_score', methods=['POST'])
def edit_score():
    data = request.get_json()
    if not data or "Sentences" not in data:
        return jsonify({"error": "Missing 'Sentences' field in JSON"}), 400

    try:
        result = generate_edit_sequences_and_score(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Route 4: Scoring Step (Pass/Fail) ---
@app.route('/decision', methods=['POST'])
def decision():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    score = data.get("Score")
    age = data.get("Age")
    percentile = data.get("Percentile", 5)

    try:
        result = evaluate_rsr_result(score, age, percentile)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Route 5: Master Endpoint ---
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'wav_file' not in request.files:
        return jsonify({"error": "Missing WAV file"}), 400

    wav_file = request.files['wav_file']
    age = request.form.get("Age")
    percentile = request.form.get("Percentile", 5)
    gt_text = request.form.get("Ground Truth", None)

    if wav_file.filename == '':
        return jsonify({"error": "No WAV file selected"}), 400

    try:
        age = int(age)
        percentile = int(percentile)
    except (ValueError, TypeError):
        return jsonify({"error": "Age and percentile must be integers"}), 400

    wav_path = os.path.join(UPLOAD_FOLDER_WAV, wav_file.filename)
    wav_file.save(wav_path)

    result = run_full_rsr_analysis(
        wav_path,
        age_in_months=age,
        percentile=percentile,
        ground_truth_text=gt_text
    )

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

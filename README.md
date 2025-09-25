# AutoRSR

A program that automates transcription, alignment, error analysis, and scoring of speech responses, designed for Redmond Sentence Recall (RSR). However, this is easily adaptable for use in other sentence recall tasks.

It uses WhisperX for transcription, custom fuzzy alignment and A*-based edit distance for error analysis, and age/percentile-based thresholds for pass/fail decisions.

---

## 📚 Table of Contents

- [Prerequisites](#prerequisites)  
- [Setup](#setup)  
- [Configuration](#configuration)  
- [Running AutoRSR](#running-autorsr)  
- [Endpoints](#endpoints)  
- [Example Usage](#example-usage)  
- [Project Structure](#project-structure)  

---

## 🧾 Prerequisites

- Python 3.10+
- CUDA-enabled GPU (recommended for WhisperX)
- `ffmpeg` installed and in `PATH`
- `venv` or `conda` enviroment (Highly Recommended)

---

## ⚙️ Setup
Setup up a venv or conda enviroment, then:
```bash
git clone https://github.com/pyraxal/AutoRSR-NoBert
pip install -r requirements.txt
```

Ensure you have `english.json` (used for normalization) placed at:

```
/auto_rsr/english.json
```

If needed, adjust the path in the `normalize()` function.

### **Note**

- Using this version of WhisperX, there is a chance that the asr.py within the WhisperX install does not contain certain important lines. If you get the error:
```
TypeError: <lambda>() missing 3 required positional arguments: 'max_new_tokens', 'clip_timestamps', and 'hallucination_silence_threshold'
```

Please find asr.py within the WhisperX package and ensure that at line 322 that after "suppress_numerals": False, you have the following three lines:
```
"max_new_tokens": None, "clip_timestamps": None, "hallucination_silence_threshold": None,
```
If not, add them, save, and it should run.

---

## 🛠 Running AutoRSR
There are **two** main ways you can use AutoRSR. You can use it locally by calling the api functions within `auto_rsr.py` like so:

    ```
    import json
    result = run_full_rsr_analysis("wavfile_location", age_in_months=65, percentile=5, "gt_location")
    print(json.dumps(result, indent=1))
    ```

Or, you can alternatively use the API. 

```
python app.py
```

It runs locally at:

```
http://127.0.0.1:5000/
```


### **Note**

- Uploaded audio and text files are stored in:
  ```
  uploads/wav_file/
  uploads/text_file/
  ```
  
- Modify `UPLOAD_FOLDER_WAV` or `UPLOAD_FOLDER_TEXT` in `app.py` if needed.



## 🌐 Endpoints

### 🔊 `POST /transcribe`

**Input**: Form-data (Multipart) with a `.wav` or `.mp3` file  
**Output**: Transcribed text in JSON format

```json
{
  "Transcription": "the black ball washed onto the blue ocean. ..."
}
```

---

### 🧩 `POST /align`

**Input**:  
```json
{
  "Transcription": "...",
  "Ground Truth": "Sentence 1\nSentence 2\nSentence 3"
}
```

(If `"Ground Truth"` is omitted, default GT is used)

**Output**:
```json
{
  "Sentences": [
    {
      "id": "1",
      "Ground Truth": "...",
      "Aligned": "..."
    },
    ...
  ]
}
```

---

### ✏️ `POST /edit_score`

**Input**:  
```json
{
  "Sentences": [
    {
      "id": "1",
      "Ground Truth": "...",
      "Aligned": "..."
    }
  ]
}
```

**Output**:  
```json
{
  "Sentences": [
    {
      "id": "1",
      "Sentence": "...",
      "Errors": 2,
      "Score": 1,
      "Edit Script": {
        "Insertions": [],
        "Deletions": [...],
        "Substitutions": [...],
        "Swaps": [...]
      }
    }
  ],
  "Total Score": 3
}
```

---

### ✅ `POST /decision`

**Input**:
```json
{
  "Score": 5,
  "Age": 65,
  "Percentile": 5
}
```

**Output**:
```json
{ "result": "Pass" }
```

---

### 🧠 `POST /analyze` (Master Pipeline)

**Form Data**:
- `wav_file`: audio file  
- `Age`: integer (in months)  
- `Percentile`: optional (default: 5)  
- `Ground Truth`: optional (one sentence per line)

**Output**:
```json
{
  "Transcription": "...",
  "Alignment": {
    "Sentences": [
      {
        "id": "...",
        "Ground Truth": "...",
        "Aligned": "..."
      },
      ...
    ]
  },
  "Edit and Score": {
    "Sentences": [
      {
        "id": "...",
        "Sentence": "...",
        "Errors": ...,
        "Score": ...,
        "Edit Script": {
          "Insertions": [...],
          "Deletions": [...],
          "Substitutions": [...],
          "Swaps": [...]
        }
      },
      ...
    ],
    "Total Score": ...
  },
  "Decision": {
    "Total Score": ...,
    "Percentile": ...,
    "Age (months)": ...,
    "Result": "Pass" // or "Fail"
  }
}
```

---

## 🧪 Usage

### **No Setup**
Visit: [https://autorsr.xlabub.com/home/](https://autorsr.xlabub.com/home/)

### **Manual**
You can use it locally by calling the api functions within auto_rsr.py like so:

```
import json
result = run_full_rsr_analysis("wavfile_location", age_in_months=65, percentile=5, "gt_location")
print(json.dumps(result, indent=1))
```
### **API**

Once the API is running, you can interact with it through curl:

**Manual Curl Commands**

```
# Transcription

curl -X POST \
  -F "wav_file=@RSR_0001.WAV;type=audio/wav" \
  http://localhost:5000/transcribe
```

```
# Alignment

curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"Transcription":"...", "Ground Truth":"Sentence1\nSentence2"}' \
  http://localhost:5000/align
```

```
# Edit/Scoring

curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"Sentences":[{"id":"1","Ground Truth":"...", "Aligned":"..."}]}' \
  http://localhost:5000/edit_score
```

```
# Pass/Fail Decision

curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"Score":5, "Age":65, "Percentile":5}' \
  http://localhost:5000/decision
```

```
# Full Pipeline

curl -X POST \
  -F "wav_file=@yourfile.wav" \
  -F "Age=65" \
  -F "Percentile=5" \
  -F $'Ground Truth=Sentence1\nSentence2' \
  http://localhost:5000/analyze
```
---

## 📁 Project Structure

```
.
├── auto_rsr.py              # Core logic (transcription, alignment, scoring)
├── app.py                   # Flask API server
├── requirements.txt         # Python dependencies
├── README.md                # Readme
└── uploads/
    ├── wav_file/            # Audio Uploads
    └── text_file/           # Text Uploads
```



import ftfy
import re
import json
import itertools
import difflib
import heapq
import os
import whisperx
from transformers.models.whisper import english_normalizer
import shutil

def normalize(target):  
    """
    Apply English text normalization to the given string using the Whisper normalizer.

    This function loads a normalization configuration from a JSON file, then
    applies a series of rules such as expanding abbreviations, fixing casing,
    and handling punctuation. 
    
    Please ensure that that the json file (english.json) is present with valid path. 

    Args:
        target (str): A raw text string to normalize.

    Returns:
        str: The normalized version of the input string.
    """
    normalizer = english_normalizer.EnglishTextNormalizer(json.loads(open("english.json").read()))
    return(normalizer(target))

# --- Transcription ---
def transcribe(filename):
    """
    Transcribe an audio file using WhisperX.

    By default, this loads the large WhisperX model on GPU, which you can change 
    by modifying the the load model line. This function also does some minor cleaning
    to standarize periods and general punctuation. 

    Args:
        filename (str): Path to the audio file to transcribe.

    Returns:
        str: Transcription
    """
    # Transcribes using whisperx
    device = "cuda"
    audio_file = filename
    batch_size = 16  # reduce if low on GPU mem
    compute_type = "float16"  # or "int8" if low on GPU mem (may reduce accuracy)

    model = whisperx.load_model("large-v3", device, compute_type=compute_type, language="en")
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)

    segments = result["segments"]
    texts = [seg["text"].strip() for seg in segments if "text" in seg]

    # Fix missing terminal periods
    fixed_texts = []
    for curr, nxt in zip(texts, texts[1:] + [""]):
        # if curr doesn't end with . ! ? and next starts with uppercase letter
        if not re.search(r"[\.!\?]$", curr) and re.match(r"^[A-Z]", nxt):
            curr = curr + "."
        fixed_texts.append(curr)
    # join all (the dummy "" at end does no harm)
    return " ".join(fixed_texts).strip()

# --- Text standardization ---
def standarize(a_str):
    """
    Clean and normalize arbitrary text for comparison and scoring.

    Steps:
      1. This fixes unicode (ftfy).
      2. Strip extra whitespace and filler tokens (e.g., 'uh', 'um').
      3. Normalizes with above function for consistent casing and punctuation.

    Args:
        a_str (str): An input text string.

    Returns:
        str: Standarized string
    """
    a_str = ftfy.fix_text(a_str)
    a_str = a_str.strip()
    a_str = re.sub(r"\b(?:uh|um|ah|)\b", '', a_str, flags=re.IGNORECASE)
    a_str = re.sub(r"\s+", ' ', a_str).strip()
    a_str = normalize(a_str)
    return a_str

# --- Scoring with Transcriptions ---
def score_rsr_errors(response, ground_truth):
    """
    Compute the minimal edit script between two word sequences.

    Uses A* search over insert, delete, substitute, and swap operations to
    find a sequence of edits that transforms `response` into `ground_truth`.

    Args:
        response (List[str]): A list containing a list of words
        ground_truth (List[str]): Ground Truth list also containing a list of words

    Returns:
        dict: A breakdown of Insertions, Deletions, Substitutions, and Swaps needed.
        Dictionary contains: 
        "Edit Script": {'Insertions': [], 'Deletions': [], 'Substitutions': [], 'Swaps': []}
    """
    goal = tuple(ground_truth)
    start = tuple(response)

    def h(state):
        return sum(1 for a, b in zip(state, goal) if a != b) + abs(len(state) - len(goal))

    open_set = [(h(start), 0, start)]
    parent = {start: None}
    parent_op = {start: None}
    cost_so_far = {start: 0}

    while open_set:
        _, g, state = heapq.heappop(open_set)
        if state == goal:
            # backtrack gather ops
            ops = []
            cur = state
            while parent_op[cur] is not None:
                ops.append(parent_op[cur])
                cur = parent[cur]
            ops.reverse()
            # group by type
            out = {'Insertions': [], 'Deletions': [], 'Substitutions': [], 'Swaps': []}
            for op in ops:
                typ = op[0]
                if typ == 'insert':
                    _, pos, tok = op
                    out['Insertions'].append((pos, tok))
                elif typ == 'delete':
                    _, pos, tok = op
                    out['Deletions'].append((pos, tok))
                elif typ == 'substitute':
                    _, pos, old, new = op
                    out['Substitutions'].append((pos, old, new))
                elif typ == 'swap':
                    _, i, j, wi, wj = op
                    out['Swaps'].append((i, j, wi, wj))
            return out

        L = len(state)
        # 1) transpose any two
        for i in range(L):
            for j in range(i+1, L):
                wi, wj = state[i], state[j]
                new = list(state)
                new[i], new[j] = wj, wi
                neigh = tuple(new)
                op = ('swap', i, j, wi, wj)
                new_cost = g + 1
                if new_cost < cost_so_far.get(neigh, float('inf')):
                    cost_so_far[neigh] = new_cost
                    parent[neigh] = state
                    parent_op[neigh] = op
                    heapq.heappush(open_set, (new_cost + h(neigh), new_cost, neigh))

        # 2) insert any target token
        for i in range(L+1):
            for tok in set(goal):
                new = state[:i] + (tok,) + state[i:]
                neigh = new
                op = ('insert', i, tok)
                new_cost = g + 1
                if new_cost < cost_so_far.get(neigh, float('inf')):
                    cost_so_far[neigh] = new_cost
                    parent[neigh] = state
                    parent_op[neigh] = op
                    heapq.heappush(open_set, (new_cost + h(neigh), new_cost, neigh))

        # 3) delete
        for i in range(L):
            tok = state[i]
            new = state[:i] + state[i+1:]
            neigh = new
            op = ('delete', i, tok)
            new_cost = g + 1
            if new_cost < cost_so_far.get(neigh, float('inf')):
                cost_so_far[neigh] = new_cost
                parent[neigh] = state
                parent_op[neigh] = op
                heapq.heappush(open_set, (new_cost + h(neigh), new_cost, neigh))

        # 4) substitute
        for i in range(L):
            for tok in set(goal):
                if state[i] != tok:
                    old = state[i]
                    new = state[:i] + (tok,) + state[i+1:]
                    neigh = new
                    op = ('substitute', i, old, tok)
                    new_cost = g + 1
                    if new_cost < cost_so_far.get(neigh, float('inf')):
                        cost_so_far[neigh] = new_cost
                        parent[neigh] = state
                        parent_op[neigh] = op
                        heapq.heappush(open_set, (new_cost + h(neigh), new_cost, neigh))

    return {'Insertions': [], 'Deletions': [], 'Substitutions': [], 'Swaps': []}

# --- Preprocessing + Alignment---
def fuzzy_align_response_to_gt(gt, resp, default_thresh=60.0, min_words=2):
    """
    Find the snippet of model transcription that best matches a ground truth sentence.

    Splits everything into sentences, then scores each sentence of the resp by its
    similarity to each sentence to the ground truth. If the best candidate is 
    significantly longer than the target, then the best canadate will be matched 
    with word level alignment to ensure there are not multiple sentences in one sentence. 
    If it isn't, it will use default judgement which is to match the response with the 
    ground truth that has the greatest similarity beyond the default threshold inputted. 
    Otherwise, it wont be matched with anything and will be matched with an empty string. 

    Args:
        gt (str): The ground truth sentence.
        resp (str): The full model transcription.
        default_thresh (float): Minimum similarity percent to accept a match.
        min_words (int): Minimum number of words for a sentence candidate.

    Returns:
        str: The aligned sentence fragment from `resp` that best matches `gt`.
    """
    def normalize_txt(text):
        t = text.lower()
        t = re.sub(r"[^\w\s]", "", t)
        return re.sub(r"\s+", " ", t).strip()

    def sim(a, b):
        return difflib.SequenceMatcher(None, a, b).ratio() * 100

    def find_best_alignment(paragraph, target_sentence):
        paragraph_words = paragraph.split()
        target_words = target_sentence.split()
        n, m = len(paragraph_words), len(target_words)

        def compute_alignment(segment, target):
            dp = [[0] * (len(target) + 1) for _ in range(len(segment) + 1)]
            for i in range(1, len(segment) + 1):
                dp[i][0] = i
            for j in range(1, len(target) + 1):
                dp[0][j] = j
            for i in range(1, len(segment) + 1):
                for j in range(1, len(target) + 1):
                    match_cost = 0 if segment[i - 1] == target[j - 1] else 1
                    dp[i][j] = min(
                        dp[i - 1][j - 1] + match_cost,
                        dp[i - 1][j] + 1,
                        dp[i][j - 1] + 1
                    )
            return dp[-1][-1]

        min_distance = float('inf')
        best_segment = None

        for start in range(n - m + 1):
            segment = paragraph_words[start:start + m]
            distance = compute_alignment(segment, target_words)
            if distance < min_distance:
                min_distance = distance
                best_segment = segment

        def trim_segment(segment, target):
            while len(segment) > 1 and compute_alignment(segment[1:], target) <= compute_alignment(segment, target):
                segment = segment[1:]
            while len(segment) > 1 and compute_alignment(segment[:-1], target) <= compute_alignment(segment, target):
                segment = segment[:-1]
            return segment

        if best_segment:
            best_segment = trim_segment(best_segment, target_words)
            return " ".join(best_segment)
        else:
            return ""

    # Normalize and prepare
    gt_norm = normalize_txt(gt)
    gt_len = len(gt_norm.split())
    thresh = 40.0 if gt_len < 5 else default_thresh

    # Split into sentence candidates
    cands = re.split(r'(?<=[.!?])\s+', resp)
    cands = [c for c in cands if len(c.split()) >= min_words]

    best, best_score = "", 0.0
    for cand in cands:
        cand_n = normalize_txt(cand)
        score = sim(cand_n, gt_norm)
        if score > best_score:
            best_score, best = score, cand

    if best_score >= thresh:
        best_norm = normalize_txt(best)
        if len(best_norm.split()) > gt_len * 1.3:
            refined = find_best_alignment(best_norm, gt_norm)
            return refined.strip()
        return best.strip()
    else:
        return ""

def word_list(sentence):
    """
    Split a sentence into words based on whitespace.

    Args:
        sentence (str): Input text.

    Returns:
        List[str]: Tokenized word list.
    """
    return sentence.split()

# Scoring
def prepare(gt, response):
    """
    Helper function for api input. It helps convert the json input to something workable.

    Args:
        gt (str): JSON string with a top-level 'Sentences' list.
        response (List[str]): List of IDs to include.

    Returns:
        Tuple[List[str], List[str]]: Parallel lists of ground truth and model responses.
    """
    data = json.loads(gt)
    ids = set(int(i) for i in response)
    sents = [e for e in data.get('Sentences', []) if int(e['id']) in ids]
    gt = [e['Ground Truth'] for e in sents]
    resp = [e['Response'] for e in sents]
    return gt, resp

#matches and returns errors of multiple (or single) gt to response
def batch_rsr(gt, responses):
    """
    Batch process multiple sentences. This will align, standardize, and score errors.

    For each ID in `responses`, finds the matching transcription snippet,
    standardizes both GT and response, then computes edit scripts and scores.

    Args:
        gt (str): JSON string of ground truth sentences.
        responses (List[str]): List of string IDs to evaluate.

    Returns:
        Tuple[List[dict], List[int]]: Error scripts and numeric scores per sentence.
    """
    ground_truth, responses = prepare(gt, responses)
    ground_truth = [standarize(s) for s in ground_truth]

    responses = [
        standarize(fuzzy_align_response_to_gt(gt, resp))
        for gt, resp in zip(ground_truth, responses)
    ]

    all_scripts = []
    all_scores  = []

    for tgt_sent, resp_sent in zip(ground_truth, responses):
        gt  = word_list(tgt_sent)
        res = word_list(resp_sent)

        print(res)
        print(gt)
        ops_dict = score_rsr(res, gt)
        score = sum(len(v) for v in ops_dict.values())

        all_scripts.append(ops_dict)
        all_scores.append(score)

    return all_scripts, all_scores

#takes total rsr score, age in months, and percentile
def rsr_pass_fail(total_score, age, percentile=5):
    """
    Determine pass/fail based on total RSR score, age in months, and percentile cutoffs.

    Uses predefined thresholds that vary by age (in years and months) and percentile.

    Args:
        total_score (int or float): Cumulative error-based score from RSR.
        age (int or str): Participant age in months.
        percentile (int): Desired percentile cutoff (e.g., 5, 10, 15). Default 5. 

    Returns:
        str: 'Pass', 'Fail', or 'N/A' if out of supported age range.
    """
    try:
        month = int(age)
        total_score = float(total_score)
    except ValueError:
        return "N/A"

    if month < 0:
        return "N/A"

    result = ""
    age = month // 12
    month = month % 12

    if age == 5:
        if 0 <= month <= 5:
            if percentile == 15 and total_score > 8.9:
                result = "Pass"
            elif percentile == 10 and total_score > 5.0:
                result = "Pass"
            elif percentile == 5 and total_score > 1.3:
                result = "Pass"
            else:
                result = "Fail"
        elif 6 <= month <= 11:
            if percentile == 15 and total_score > 9.0:
                result = "Pass"
            elif percentile == 10 and total_score > 8.0:
                result = "Pass"
            elif percentile == 5 and total_score > 3.95:
                result = "Pass"
            else:
                result = "Fail"

    elif age == 6:
        if 0 <= month <= 5:
            if percentile == 15 and total_score > 10.05:
                result = "Pass"
            elif percentile == 10 and total_score > 9.7:
                result = "Pass"
            elif percentile == 5 and total_score > 6.0:
                result = "Pass"
            else:
                result = "Fail"
        elif 6 <= month <= 11:
            if percentile == 15 and total_score > 16.0:
                result = "Pass"
            elif percentile == 10 and total_score > 14.0:
                result = "Pass"
            elif percentile == 5 and total_score > 12.0:
                result = "Pass"
            else:
                result = "Fail"

    elif age == 7:
        if 0 <= month <= 5:
            if percentile == 15 and total_score > 13.25:
                result = "Pass"
            elif percentile == 10 and total_score > 11.5:
                result = "Pass"
            elif percentile == 5 and total_score > 5.25:
                result = "Pass"
            else:
                result = "Fail"
        elif 6 <= month <= 11:
            if percentile == 15 and total_score > 19.0:
                result = "Pass"
            elif percentile == 10 and total_score > 16.0:
                result = "Pass"
            elif percentile == 5 and total_score > 14.1:
                result = "Pass"
            else:
                result = "Fail"

    elif age == 8:
        if 0 <= month <= 5:
            if percentile == 15 and total_score > 21.75:
                result = "Pass"
            elif percentile == 10 and total_score > 18.5:
                result = "Pass"
            elif percentile == 5 and total_score > 15.25:
                result = "Pass"
            else:
                result = "Fail"
        elif 6 <= month <= 11:
            if percentile == 15 and total_score > 20.1:
                result = "Pass"
            elif percentile == 10 and total_score > 18.4:
                result = "Pass"
            elif percentile == 5 and total_score > 15.7:
                result = "Pass"
            else:
                result = "Fail"

    elif age == 9:
        if 0 <= month <= 5:
            if percentile == 15 and total_score > 23.05:
                result = "Pass"
            elif percentile == 10 and total_score > 23.0:
                result = "Pass"
            elif percentile == 5 and total_score > 22.35:
                result = "Pass"
            else:
                result = "Fail"

    else:
        result = "N/A"

    return result

#takes rsr errors ([2,3,4...]) and returns a score
def score_rsr(a_list):
    """
    Convert error counts into a raw RSR score.

    Assigns:
      - 2 points if zero errors,
      - 1 point if fewer than 4 errors,
      - 0 points otherwise.

    Args:
        a_list (List[int]): List of error counts to score.

    Returns:
        int: Sum of points across all items.
    """
    sum = 0    
    for element in a_list:
        if element == 0:
            sum+=2
        elif element < 4:
            sum+=1    
    return sum

###############################################################################
###############################################################################
#api stuff
#look at api_interface for detailed documentation as to what format everything expects
def transcribe_to_json(mp3_file_path):
    """
    Convenience wrapper: take an MP3 path and return a JSON "Transcription" object.

    Args:
        mp3_file_path (str): Path to the .mp3 audio file.

    Returns:
        dict: {'Transcription': full_text}
    """
    transcription = transcribe(mp3_file_path)
    return {"Transcription": transcription}

def align_transcription_to_ground_truth(transcription_data, ground_truth_text=None):
    """
    Aligns transcription sentences to ground truth sentences.
    If no custom GT text is provided, falls back to a default list of example sentences
    
    Args:
        transcription_data (dict): {"Transcription": "..."}
        ground_truth_text (str or None): Optional multiline string with one GT sentence per line.
    
    Returns:
        dict: {"Sentences": [{"id": "...", "Ground Truth": "...", "Aligned": "..."}]}
    """
    if "Transcription" not in transcription_data:
        raise ValueError("Missing 'Transcription' in input data.")

    # Use default GT if not provided
    if ground_truth_text is None:
        ground_truth_text = """The big football player washed the car with the hose.
        All of the pictures were colored by his little sister.
        The rose bushes were planted yesterday by the girl scouts.
        The happy little girl kicked the ball over the fence.
        His little brother cleaned the dirty dishes and cups.
        A special cage was made to hold the dangerous animals.
        Everybody in my school colored Easter eggs for the picnic.
        A new hole was dug for the kid's swimming pool.
        Only the first graders made a birdhouse for their parents.
        My little sister's dog caught the ball on the first bounce.
        The soccer ball was kicked into the school's parking lot.
        The lion's teeth were cleaned with a giant toothbrush.
        Some of the kids dug holes in the sand two feet deep.
        The little white mouse was caught by our neighbor's cat.
        The second grade students planted coconuts in the garden.
        The dirty clothes were washed with soap one more time."""

    # Process GT into lines
    gt_lines = [line.strip() for line in ground_truth_text.strip().split('\n') if line.strip()]
    transcription = transcription_data["Transcription"]
    print(transcription)
    print(gt_lines)

    output = {"Sentences": []}
    for idx, gt in enumerate(gt_lines, start=1):
        aligned = fuzzy_align_response_to_gt(gt, transcription)
        output["Sentences"].append({
            "id": str(idx),
            "Ground Truth": gt,
            "Aligned": aligned
        })
    
    return output

def generate_edit_sequences_and_score(aligned_data):
    """
    Takes aligned transcription data and returns scored errors and edit sequences,
    including a total score across all sentences.
    
    Args:
        aligned_data (dict): {
            "Sentences": [
                {"id": "...", "Ground Truth": "...", "Aligned": "..."},
                ...
            ]
        }

    Returns:
        dict: {
            "Total Score": int,
            "Sentences": [
                {
                    "id": "...",
                    "Sentence": "...",
                    "Errors": int,
                    "Score": int,
                    "Edit Script": {...}
                },
                ...
            ]
        }
    """
    output = {"Sentences": []}
    total_score = 0

    for item in aligned_data.get("Sentences", []):
        sent_id = item["id"]
        gt = standarize(item["Ground Truth"])
        aligned = standarize(item["Aligned"])

        gt_words = gt.split()
        aligned_words = aligned.split()

        # Get error script
        edits = score_rsr_errors(aligned_words, gt_words)
        error_count = sum(len(v) for v in edits.values())
        score = score_rsr([error_count])
        total_score += score

        output["Sentences"].append({
            "id": sent_id,
            "Sentence": aligned,
            "Errors": error_count,
            "Score": score,
            "Edit Script": edits
        })

    output["Total Score"] = total_score
    return output


def evaluate_rsr_result(score, age, percentile=5):
    """
    Takes a score (int, float, or string), age in months, and percentile,
    and returns "Pass", "Fail", or "N/A".
    
    Args:
        score (str or float or int): Total RSR score
        age (str or int): Age in months
        percentile (int): Percentile cutoff (default = 5)
    
    Returns:
        str: "Pass", "Fail", or "N/A"
    """
    try:
        score = float(score)
        age = int(age)
    except (ValueError, TypeError):
        return "Incorrect Paremeters"

    if age < 0:
        return "N/A"

    return rsr_pass_fail(score, age, percentile)

def run_full_rsr_analysis(mp3_path, age_in_months, percentile=5, ground_truth_text=None):
    """
    Master pipeline: runs transcription → alignment → edit scoring → pass/fail decision.
    
    Args:
        mp3_path (str): Path to the .mp3 file
        age_in_months (int or str): Age of participant in months
        percentile (int): Percentile for scoring thresholds (default = 5)
        ground_truth_text (str or None): Optional ground truth (1 sentence per line)

    Returns:
        dict: Full result including all intermediate steps and final decision
    """
    # Step 1: Transcription
    transcription_data = transcribe_to_json(mp3_path)

    # Step 2: Alignment
    alignment_data = align_transcription_to_ground_truth(transcription_data, ground_truth_text)

    # Step 3: Edit/Scoring Step
    edit_data = generate_edit_sequences_and_score(alignment_data)

    # Step 4: Pass/Fail
    total_score = edit_data.get("Total Score", 0)
    pass_fail_result = evaluate_rsr_result(total_score, age_in_months, percentile)

    # Bundle all results
    return {
        "Transcription": transcription_data["Transcription"],
        "Alignment": alignment_data,
        "Edit and Score": edit_data,
        "Decision": {
            "Total Score": total_score,
            "Percentile": percentile,
            "Age (months)": age_in_months,
            "Result": pass_fail_result
        }
    }


#Example
'''
import json
result = run_full_rsr_analysis("wavfile_location", age_in_months=65, percentile=5, "gt_location")
print(json.dumps(result, indent=1))
'''

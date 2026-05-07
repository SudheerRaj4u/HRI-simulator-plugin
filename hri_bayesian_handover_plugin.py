"""
EE653 Human-Robot Interaction — Final Project Plugin
=====================================================
Author  : Sudheer Raj
Module  : EE653 Human-Robot Interaction
Option  : Pre-made Simulator (Simulated Perception)

Title   : Bayesian Goal-Aware Handover with Gesture and Spatial Intent Fusion

Overview
--------
This plugin implements a full HRI perception → inference → action pipeline:

  1. Perception      – MediaPipe hand landmarks (gestures + fingertip position)
  2. Inference       – Bayesian Goal Recognition updates P(G | O) each frame
  3. Temporal Filter – HMM-style prediction step keeps intent stable over time
  4. Confirmation    – Robot action is only triggered when P(G | O) > 0.8
  5. Handover        – Confirmed intent triggers the saved robot operation via id
  6. Adaptation      – Stability + cooldown prevent retriggering on noisy frames

Bayesian Goal Recognition Formula (Equation 1)
-----------------------------------------------
                  P(O | G) * P(G)
  P(G | O)  =  ---------------------
               Σ_i  P(O | G_i) * P(G_i)

  Where:
    G       = goal hypothesis (one saved robot operation)
    O       = current observation (fingertip position + gesture evidence)
    P(G)    = prior belief from previous frame (temporal continuity)
    P(O|G)  = Gaussian likelihood: exp(−d² / 2σ²) × gesture_multiplier
    P(G|O)  = posterior — displayed live on the red/green box overlay

HMM Prediction Step (Equation 2)
---------------------------------
  P(G_t) = Σ_j  P(G_t | G_{t-1}=j) * P(G_{t-1}=j | O_{1:t-1})

  Where:
    STAY_PROB   = probability that intent stays the same frame-to-frame
    SWITCH_PROB = probability that intent switches to any other goal

Confirmation Threshold
-----------------------
  Trigger robot iff  P(G* | O) > CONFIRM_THRESHOLD  (default 0.80)

Minimum-Jerk Trajectory Note
-----------------------------
  For a smooth, human-predictable robot trajectory the robot should follow:
      x(t) = x_0 + (x_f − x_0) * [10τ³ − 15τ⁴ + 6τ⁵],  τ = t / T
  This is handled by the simulator's built-in motion controller when a
  saved robot operation is triggered via trigger_operation_id.
"""

# ── IMPORTS ──────────────────────────────────────────────────────────────────
# We only need Python's built-in 'math' library for math.exp(), which computes
# the exponential function used in our Gaussian probability formula.
# The simulator provides everything else (hand data, box positions, timing).
import math

# ---------------------------------------------------------------------------
# Plugin metadata (shown in the simulator UI)
# ---------------------------------------------------------------------------
# ── PLUGIN METADATA ──────────────────────────────────────────────────────────
# This dictionary is read by the simulator when the plugin is uploaded.
# The 'name' and 'description' appear in the simulator's UI so the user
# can confirm the correct plugin is active.
PLUGIN_META = {
    "name": "Bayesian Goal-Aware Handover",
    "description": (
        "Combines gesture recognition and Bayesian spatial intent inference "
        "to infer which robot handover operation the user wants. "
        "Triggers when P(G|O) > 0.80 using Bayesian Goal Recognition."
    ),
}

# ---------------------------------------------------------------------------
# ── TUNING PARAMETERS ────────────────────────────────────────────────────────
# These are the "control knobs" of the system. Changing a number changes
# behaviour without touching any logic — like dials on a mixing board.
# ---------------------------------------------------------------------------

# SIGMA = 0.13  (the Gaussian spread, σ)
# Imagine a circle of influence drawn around each target box on screen.
# A hand within ~13% of screen width from the box centre counts as strong
# evidence. Smaller σ = tighter/more precise; larger σ = more forgiving.
SIGMA = 0.13

# INSIDE_BOX_BOOST / INSIDE_BOX_PENALTY
# When the fingertip is CLEARLY INSIDE one box:
#   → that box gets ×12  (very strong "yes" signal)
#   → the OTHER box gets ×0.07 (near-zero "no" signal)
# The contrast ratio (12 ÷ 0.07 ≈ 171×) makes intent unmistakable
# when the hand is squarely over a target.
INSIDE_BOX_BOOST = 12.0
INSIDE_BOX_PENALTY = 0.07

# GESTURE_BOOST = 2.5
# When the hand gesture (e.g. open palm) matches the type of operation
# linked to a target box, that box's evidence score is multiplied by 2.5.
# Gesture provides INDEPENDENT evidence on top of position — very useful
# when the hand hovers ambiguously between two boxes.
GESTURE_BOOST = 2.5

# HMM TRANSITION PROBABILITIES  (Equation 2 — temporal filter)
# STAY_PROB = 0.82 → 82% of the previous belief carries forward unchanged.
#                    Intent is "sticky" — it won't flip on a single noisy frame.
# SWITCH_PROB = 0.18 → 18% is spread to other goals, keeping the system
#                      open to genuine intent changes after sustained evidence.
STAY_PROB = 0.82
SWITCH_PROB = 1.0 - STAY_PROB   # the complement: probability that intent changes

# CONFIRM_THRESHOLD = 0.80  (the robot's minimum confidence before acting)
# The robot only fires when it is at least 80% sure about the human's intent.
# Lower = faster response but more false triggers.
# Higher = safer but the robot feels slower and less responsive.
CONFIRM_THRESHOLD = 0.80

# REQUIRED_STABLE_FRAMES = 2
# Even at 80%+ confidence, the SAME intent must appear in 2 consecutive
# webcam frames before triggering. This eliminates single-frame MediaPipe
# glitches at almost zero cost — at 30fps, 2 frames is only ~67 ms.
REQUIRED_STABLE_FRAMES = 2

# COOLDOWN_MS = 1400  (1.4 seconds of silence after each trigger)
# After firing, no new triggers are allowed for 1.4 s.
# Without this, while the hand stays inside the box (still confident),
# the robot would receive dozens of identical commands per second and
# loop the same motion endlessly. 1.4 s ≈ the robot's travel time.
COOLDOWN_MS = 1400

# FOUR_OPERATION_MAP — the gesture-to-operation routing table
# Maps (hand_side, gesture) → operation list index (0–3).
# Think of it as a 2×2 grid:
#   Left  + open gestures (open_palm, peace, three_fingers) → Op 0  (green)
#   Left  + pinch gestures (pinch, point, fist)             → Op 1  (red)
#   Right + open gestures                                   → Op 2  (green)
#   Right + pinch gestures                                  → Op 3  (red)
# To remap a gesture, just change the number on the right of each row.
# Key: (hand_side, gesture_name)  →  Value: index into saved_operations list
FOUR_OPERATION_MAP = {
    ("Left",  "open_palm"):   0,
    ("Left",  "peace"):       0,
    ("Left",  "three_fingers"): 0,
    ("Left",  "pinch"):       1,
    ("Left",  "point"):       1,
    ("Left",  "fist"):        1,
    ("Right", "open_palm"):   2,
    ("Right", "peace"):       2,
    ("Right", "three_fingers"): 2,
    ("Right", "pinch"):       3,
    ("Right", "point"):       3,
    ("Right", "fist"):        3,
}

# GESTURE_PRIORITY — tiebreaker when two hands show conflicting gestures
# Higher number = higher priority. "pinch" (7) beats "hover" (1) because
# a deliberate pinch is a far clearer intent signal than a vague hover.
GESTURE_PRIORITY = {
    "pinch": 7, "open_palm": 6, "peace": 5,
    "three_fingers": 4, "point": 3, "fist": 2, "hover": 1,
}

ACTIONABLE_GESTURES = frozenset(FOUR_OPERATION_MAP.values())   # operation indices
ACTIONABLE_GESTURE_NAMES = frozenset(
    g for (_, g) in FOUR_OPERATION_MAP
)

PINCH_THRESHOLD = 0.07        # normalised distance for pinch detection

# ---------------------------------------------------------------------------
# ── GLOBAL STATE — memory that persists between frames ───────────────────────
# Because process_frame() is called fresh on every webcam frame, we need
# module-level globals to remember what happened in previous frames.
#
# POSTERIOR        — current probability distribution over goals, per hand.
#                    e.g. {"Left": {"green": 0.73, "red": 0.27}}
#                    Updated every frame by the Bayes update.
#
# STABILITY        — consecutive-frame counter per (hand, intent, gesture).
#                    Resets to 0 the instant anything changes.
#
# LAST_TRIGGER_KEY — string fingerprint of the last fired trigger;
#                    used to detect whether we are still in cooldown.
#
# LAST_TRIGGER_AT  — timestamp (ms) of the last trigger for cooldown math.
# ---------------------------------------------------------------------------
POSTERIOR        = {}   # {hand_side: {box_id: probability}}
STABILITY       = {}   # {state_key: consecutive_frame_count}
LAST_TRIGGER_KEY = None # fingerprint of the last trigger that fired
LAST_TRIGGER_AT  = 0   # timestamp (ms) of the last trigger


# ===========================================================================
# ── setup() — called ONCE when the plugin is uploaded ────────────────────────
# Think of it as the "power-on" routine.
# It wipes any leftover state from a previous session and tells the simulator
# how many saved robot operations are ready to use.
# ===========================================================================
def setup(payload):
    """
    Called once when the plugin is loaded.
    Resets all global state and reports ready status to the simulator.
    """
    # Declare that we are modifying the module-level globals,
    # not creating new local variables with the same names.
    global POSTERIOR, STABILITY, LAST_TRIGGER_KEY, LAST_TRIGGER_AT

    POSTERIOR        = {}
    STABILITY        = {}
    LAST_TRIGGER_KEY = None
    LAST_TRIGGER_AT  = 0

    n_ops = len(payload.get("saved_operations", []))
    return {
        "status": "Bayesian Handover Plugin ready",
        "available_operations": n_ops,
        "confirm_threshold":    CONFIRM_THRESHOLD,
        "note": (
            "Trigger condition: P(G|O) > {:.0%}. "
            "Make sure at least 2 (ideally 4) operations are saved.".format(
                CONFIRM_THRESHOLD
            )
        ),
    }


# ===========================================================================
# UTILITY / MATH HELPERS
# Small, single-purpose functions called by the main Bayesian pipeline.
# You can think of these as the individual "building blocks" of the maths.
# ===========================================================================

# normalize()
# Rescales a dictionary of raw numbers so they all add up to exactly 1.0,
# turning them into valid probabilities.
# Example:  {"green": 3.0, "red": 1.0}  →  {"green": 0.75, "red": 0.25}
# This is the denominator (normalisation) step of Bayes' formula (Eq. 1).
def normalize(prob_dict):
    """Ensure probabilities sum to 1.0 (Equation 1 denominator)."""
    total = sum(prob_dict.values()) or 1.0
    return {k: v / total for k, v in prob_dict.items()}


# bayes_update()  —  one step of Bayes' rule  (Equation 1)
# Inputs:
#   prior      — what we believed BEFORE this frame  e.g. {"green": 0.6, "red": 0.4}
#   likelihood — how well the current observation fits each goal
# Process: multiply likelihood × prior for each goal, then normalise.
# Output:  posterior — the updated belief AFTER seeing this frame.
#
# Plain-English analogy:
#   "How likely is this hand position IF the human wants green?" ×
#   "How likely did I already think they want green?"
#   → new score for green. Repeat for red. Divide so they add up to 100%.
def bayes_update(prior, likelihood):
    """
    Apply one step of Bayes rule (Equation 1).

          posterior[G] = likelihood[G] * prior[G]   (unnormalized)
    Then normalize so all posteriors sum to 1.
    """
    # Equation 1 numerator: for each goal, multiply likelihood × prior.
    unnormalized = {g: likelihood[g] * prior[g] for g in prior}
    return normalize(unnormalized)


# hmm_predict()  —  HMM temporal smoothing step  (Equation 2)
# Called at the START of every frame, BEFORE the Bayes update.
# It slightly "blurs" the previous belief toward uncertainty so the system
# stays open to intent changes rather than locking in forever on one goal.
#
# With STAY_PROB = 0.82:
#   If we were 90% confident about green last frame, we enter this frame
#   at ~83% — still confident, but new evidence can shift things.
#
# Plain-English: "I was pretty sure you wanted the green box last frame,
#   but I'll keep a small open mind in case you've changed your mind."
def hmm_predict(previous):
    """
    HMM prediction step (Equation 2).

    Mixes the previous posterior with a flat transition distribution
    so that intent belief decays toward uniform when evidence is absent.
    Supports any number of hypotheses (goals).
    """
    n = len(previous)
    if n == 0:
        return previous
    uniform_share = SWITCH_PROB / n       # probability mass shared to all states
    return {
        g: STAY_PROB * previous[g] + uniform_share
        for g in previous
    }


# gaussian_likelihood()  —  spatial evidence  P(O | G)
# Answers: "If the human truly wants goal G (a box at a known screen
# position), how likely is it that their fingertip appears HERE?"
#
# Uses a 2-D Gaussian bell curve centred on the box centre:
#   likelihood = exp( −distance² / (2 × σ²) )
#   • Fingertip AT box centre  → likelihood = 1.0  (maximum evidence)
#   • Fingertip FAR away       → likelihood → 0.0  (negligible evidence)
# The bell-curve shape gives a smooth, physics-inspired falloff.
def gaussian_likelihood(point, center):
    """
    P(O | G) — Gaussian spatial likelihood.
    Returns exp(−d² / 2σ²) where d is the Euclidean distance
    between the observed fingertip position and the target center.
    """
    dx = point["x"] - center["x"]
    dy = point["y"] - center["y"]
    d_sq = dx * dx + dy * dy
    return math.exp(-d_sq / (2.0 * SIGMA * SIGMA))


# inside_box()  —  simple geometric rectangle check
# Returns True if the fingertip/palm point falls inside the given box.
# Used to decide when to apply INSIDE_BOX_BOOST (×12) or INSIDE_BOX_PENALTY (×0.07).
def inside_box(point, box):
    """Return True if the fingertip/palm point is inside the given box."""
    return (
        box["x"] <= point["x"] <= box["x"] + box["width"]
        and box["y"] <= point["y"] <= box["y"] + box["height"]
    )


# box_centers()  —  extracts target box geometry from the frame data
# The simulator provides coloured target boxes (red, green, …).
# This helper pulls out each box's centre point (for the Gaussian distance
# calculation) and the full bounding rectangle (for the inside_box check).
def box_centers(frame):
    """
    Extract target box centers and raw box data from frame["bayes_boxes"].
    Returns (centers_dict, boxes_dict) keyed by box id ("red", "green", …).
    """
    centers = {}
    boxes   = {}
    for box in frame.get("bayes_boxes", {}).get("boxes", []):
        bid = box["id"]
        centers[bid] = {
            "x": box["x"] + box["width"]  * 0.5,
            "y": box["y"] + box["height"] * 0.5,
        }
        boxes[bid] = box
    return centers, boxes


# ===========================================================================
# GESTURE CLASSIFICATION
# MediaPipe provides 21 hand-joint landmarks per hand. The functions below
# convert those raw landmarks into a human-readable gesture label.
#
# No machine learning here — pure rule-based logic:
#   • Which fingers are extended? (tip is higher on screen than its knuckle)
#   • How close is the thumb tip to the index fingertip? (pinch distance)
# ===========================================================================

# _finger_states()  —  fetches the pre-computed finger boolean flags
# Returns a dict like {"indexExtended": True, "middleExtended": False, ...}
# telling us which fingers are currently raised.
def _finger_states(hand):
    return hand.get("fingerStates") or {}


# _count_no_thumb()  —  counts raised fingers EXCLUDING the thumb (0–4)
# Used to distinguish: open palm (4), three fingers (3), point (1), fist (0).
def _count_no_thumb(hand):
    rv = hand.get("fingerCountNoThumb")
    if rv is not None:
        return int(rv)
    fs = _finger_states(hand)
    return sum(
        1 for k in ("indexExtended", "middleExtended", "ringExtended", "pinkyExtended")
        if fs.get(k)
    )


# _count_with_thumb()  —  same but includes the thumb (0–5)
# A fully open hand should score 5. Used for the open_palm check.
def _count_with_thumb(hand):
    rv = hand.get("fingerCountWithThumb")
    if rv is not None:
        return int(rv)
    fs = _finger_states(hand)
    return _count_no_thumb(hand) + (1 if fs.get("thumbExtended") else 0)


# classify_gesture()  —  THE GESTURE RECOGNISER
# Converts raw finger data into one of seven gesture labels.
# Rules are checked in priority order (most distinctive pattern first):
#
#   ┌─────────────────┬──────────────────────────────────────────────────┐
#   │ Gesture         │ Rule (plain English)                             │
#   ├─────────────────┼──────────────────────────────────────────────────┤
#   │ pinch           │ Thumb & index very close together                │
#   │ open_palm       │ All 4+ fingers + thumb extended, not pinching    │
#   │ three_fingers   │ Index, middle, ring up — pinky down              │
#   │ peace / V-sign  │ Index + middle up — ring & pinky down            │
#   │ point           │ Only index finger raised                         │
#   │ fist            │ All fingers curled, not pinching                 │
#   │ hover           │ None of the above matched                        │
#   └─────────────────┴──────────────────────────────────────────────────┘
#
# Returns: (gesture_name, confidence_score)
# Confidence is a fixed heuristic — higher = rule is more distinctive.
def classify_gesture(hand):
    """
    Classify the hand gesture and return (gesture_name, confidence).

    Recognised gestures:
        pinch         — thumb+index close together
        open_palm     — all 4+ fingers and thumb extended
        three_fingers — index, middle, ring up; pinky down
        peace         — index, middle up; ring, pinky down
        point         — only index up
        fist          — all fingers curled
        hover         — no clear gesture
    """
    fs            = _finger_states(hand)
    pinch_dist    = hand.get("pinchDistance", 1.0)
    no_thumb      = _count_no_thumb(hand)
    with_thumb    = _count_with_thumb(hand)
    index_up      = bool(fs.get("indexExtended"))
    middle_up     = bool(fs.get("middleExtended"))
    ring_up       = bool(fs.get("ringExtended"))
    pinky_up      = bool(fs.get("pinkyExtended"))

    if pinch_dist < PINCH_THRESHOLD and index_up:
        return "pinch",         0.97
    if no_thumb >= 4 and with_thumb >= 4 and pinch_dist > 0.075:
        return "open_palm",     0.95
    if index_up and middle_up and ring_up and not pinky_up:
        return "three_fingers", 0.91
    if index_up and middle_up and not ring_up and not pinky_up:
        return "peace",         0.89
    if index_up and not middle_up and not ring_up and not pinky_up:
        return "point",         0.87
    if with_thumb <= 1 and pinch_dist > 0.08:
        return "fist",          0.84
    return "hover",             0.55


# resolved_hand_label()  —  normalises the hand-side string
# The simulator may report "left", "LEFT", or "Left".
# .title() converts any variant to "Left" / "Right" so the
# FOUR_OPERATION_MAP lookup always finds the correct entry.
def resolved_hand_label(hand):
    raw = hand.get("handedness") or hand.get("viewerSide") or "Unknown"
    # Normalise to title-case so "left", "LEFT", "Left" all match the map
    return raw.strip().title()


# ===========================================================================
# OPERATION LOOKUP
# These helpers translate an abstract Bayesian intent (which box, which hand)
# into a concrete saved robot operation the simulator can execute.
# ===========================================================================

# operation_by_index()  —  safe list access
# Retrieves saved_operations[index]. Returns None — not a crash — if the
# index is out of range (e.g. fewer operations saved than expected).
def operation_by_index(frame, index):
    """Safely retrieve a saved operation by list index. Returns None if missing."""
    saved = frame.get("saved_operations", [])
    if index is None or index < 0 or index >= len(saved):
        return None
    return saved[index]


# mapped_operation_for()  —  intent → robot operation translator
# Converts (hand side, gesture, Bayesian intent) into an actual saved
# robot operation. Two modes depending on how many operations are saved:
#
#   MODE A (≥ 4 operations) — FULL mode:
#     Uses FOUR_OPERATION_MAP: hand side + gesture together select
#     one of four operations. Richest interaction model.
#
#   MODE B (≥ 2 operations) — SIMPLE mode:
#     Ignores gesture; uses Bayesian intent only.
#     green → operation 0,  red → operation 1.
#
# Returns: (operation_dict, index)  or  (None, None) if no match found.
def mapped_operation_for(frame, hand_label, gesture, intent):
    """
    Map (hand_label, gesture, intent) to a saved operation.

    Priority:
      1. Four-operation gesture map (if >= 4 saved operations exist).
      2. Two-operation intent-only fallback (green = 0, red = 1).
    Returns (operation_dict or None, index or None).
    """
    saved = frame.get("saved_operations", [])

    if len(saved) >= 4:
        idx = FOUR_OPERATION_MAP.get((hand_label, gesture))
        if idx is not None:
            return operation_by_index(frame, idx), idx

    if len(saved) >= 2:
        idx = 0 if intent == "green" else 1 if intent == "red" else None
        return operation_by_index(frame, idx), idx

    return None, None


# ===========================================================================
# STABILITY TRACKING
# Counts consecutive frames where the same (hand, intent, gesture) is active.
# The counter resets immediately the moment anything changes.
# This enforces the REQUIRED_STABLE_FRAMES guard before any trigger fires.
# ===========================================================================

# update_stability()
# `keys` — list of state-key strings for each visible hand this frame.
# Keys still active get +1. Keys that disappeared are dropped (reset to 0).
def update_stability(keys):
    """
    Increment consecutive-frame counters for each active state key.
    Keys not present in the current frame are dropped (reset to 0 implicitly).
    """
    global STABILITY
    next_counts = {}
    for key in keys:
        next_counts[key] = STABILITY.get(key, 0) + 1
    STABILITY = next_counts
    return next_counts


# ===========================================================================
# _process_hand()  —  THE BAYESIAN PIPELINE FOR ONE HAND
#
# Called once per visible hand per webcam frame. Runs all 6 steps:
#   Step 1 — Initialise a flat 50/50 prior if this hand is newly visible
#   Step 2 — HMM prediction: carry belief forward from last frame (Eq. 2)
#   Step 3 — Compute likelihood for each target box (Gaussian + boosts)
#   Step 4 — Bayes update: prior × likelihood → posterior (Eq. 1)
#   Step 5 — Pick the winning (highest-probability) goal
#   Step 6 — Map the winning goal to a saved robot operation
# ===========================================================================

def _process_hand(frame, hand, centers, boxes):
    """
    Run the full Bayesian pipeline for a single hand.

    Returns a result dict with:
        side, gesture, confidence, intent, posterior, operation,
        operation_index, inside_red, inside_green, stability_key
    """
    global POSTERIOR

    side    = resolved_hand_label(hand)
    gesture, g_conf = classify_gesture(hand)
    point   = hand.get("indexTip") or hand.get("palmCenter")

    # ── STEP 1: INITIALISE ───────────────────────────────────────────────────
    # First time we've seen this hand → start with a flat/uniform prior.
    # If there are 2 boxes: {"green": 0.5, "red": 0.5} — equal likelihood.
    # "No idea what you want yet, so I'll assume all goals are equally likely."
    if side not in POSTERIOR:
        hypotheses      = list(centers.keys())
        n               = len(hypotheses) or 1
        POSTERIOR[side] = {h: 1.0 / n for h in hypotheses}

    # ── STEP 2: TEMPORAL SMOOTHING  (Equation 2) ─────────────────────────────
    # Apply the HMM prediction step BEFORE looking at any new evidence.
    # Blends the previous posterior with a small uncertainty component so
    # the system stays open to genuine intent changes frame-to-frame.
    # `prior` is the P(G) that feeds into Bayes' formula below.
    prior = hmm_predict(POSTERIOR[side])

    # ── STEP 3: COMPUTE LIKELIHOOD  P(O | G) ─────────────────────────────────
    # For each target box: "How likely is this observation IF the human
    # truly wants this box?"  Three factors multiply together:
    #   (a) Gaussian spatial factor   — how close is the fingertip to the box?
    #   (b) Inside-box bonus/penalty  — is the hand clearly inside one box?
    #   (c) Gesture boost             — does the gesture match this box type?
    likelihood = {}
    for box_id, center in centers.items():
        l_val = gaussian_likelihood(point, center)

        # (b) INSIDE-BOX BONUS / PENALTY ─────────────────────────────────────
        # Hand clearly inside THIS box  → ×12  (very strong "yes" signal)
        # Hand clearly inside OTHER box → ×0.07 (near-zero "no" signal)
        in_box  = inside_box(point, boxes[box_id])
        out_ids = [b for b in boxes if b != box_id]
        out_box = any(inside_box(point, boxes[b]) for b in out_ids)

        if in_box and not out_box:
            l_val *= INSIDE_BOX_BOOST    # ×12: hand is squarely on this box
        elif out_box and not in_box:
            l_val *= INSIDE_BOX_PENALTY  # ×0.07: hand is on the OTHER box

        # (c) GESTURE EVIDENCE BOOST ──────────────────────────────────────────
        # Even-index operations (0, 2) → green box
        # Odd-index  operations (1, 3) → red box
        # If the detected gesture maps to an operation that aligns with
        # THIS box, multiply its likelihood by GESTURE_BOOST (×2.5).
        op_idx = FOUR_OPERATION_MAP.get((side, gesture))
        if op_idx is not None:
            saved = frame.get("saved_operations", [])
            # green = even indices (0, 2), red = odd indices (1, 3)
            if (box_id == "green" and op_idx % 2 == 0) or \
               (box_id == "red"   and op_idx % 2 == 1):
                l_val *= GESTURE_BOOST

        # Clamp to tiny positive value — zero likelihood would permanently
        # kill a hypothesis and can never be recovered by future evidence.
        likelihood[box_id] = max(l_val, 1e-9)

    # ── STEP 4: BAYES UPDATE  (Equation 1) ───────────────────────────────────
    # prior × likelihood for each goal, then normalise so they sum to 1.
    # This is the live probability shown in the debug panel (e.g. P(green)=0.83).
    # Store it back in POSTERIOR so next frame's HMM step can use it.
    posterior       = bayes_update(prior, likelihood)
    POSTERIOR[side] = posterior

    # ── STEP 5: PICK THE WINNER ───────────────────────────────────────────────
    # The goal with the highest posterior IS the current intent.
    # `confidence` is that probability value (e.g. 0.83 = 83% sure).
    # Also record whether the fingertip is physically inside either box —
    # used in the trigger decision gate in process_frame().
    intent     = max(posterior, key=posterior.get)
    confidence = posterior[intent]

    inside_red   = inside_box(point, boxes.get("red",   {})) if "red"   in boxes else False
    inside_green = inside_box(point, boxes.get("green", {})) if "green" in boxes else False

    # ── STEP 6: OPERATION LOOKUP ──────────────────────────────────────────────
    # Convert (hand side + gesture + Bayesian intent) into an actual saved
    # robot operation from the simulator's saved_operations list.
    # Returns (None, None) safely if no matching operation exists.
    operation, op_idx = mapped_operation_for(frame, side, gesture, intent)

    return {
        "side":           side,
        "gesture":        gesture,
        "gesture_conf":   g_conf,
        "intent":         intent,
        "confidence":     confidence,
        "posterior":      posterior,
        "prior":          prior,
        "operation":      operation,
        "operation_index": op_idx,
        "inside_red":     inside_red,
        "inside_green":   inside_green,
        "stability_key":  f"{side}:{intent}:{gesture}",
    }


# ===========================================================================
# process_frame — called every frame by the simulator
# ===========================================================================

# ===========================================================================
# process_frame()  —  MAIN ENTRY POINT  (called ~30× per second)
#
# The simulator calls this on every webcam frame. `frame` contains:
#   • hands            — list of detected hands with landmarks & gesture data
#   • bayes_boxes      — the coloured target boxes (red, green) on screen
#   • saved_operations — robot operations the user has previously recorded
#   • timestamp_ms     — current time in milliseconds
#
# Returns a dict the simulator uses to:
#   (a) Draw the live debug overlay (posteriors, confidence, status text)
#   (b) Fire a robot operation via `trigger_operation_id`
# ===========================================================================
def process_frame(frame):
    """
    Main entry point called by the simulator on every webcam frame.

    Pipeline:
        Perception -> Inference (Bayes + HMM) -> Confirmation (>0.80) -> Trigger
    """
    global LAST_TRIGGER_AT, LAST_TRIGGER_KEY, POSTERIOR, STABILITY

    # ── STEP 0: READ THE SCENE ───────────────────────────────────────────────
    # Extract target-box positions and the list of visible hands from the frame.
    centers, boxes = box_centers(frame)
    hands          = frame.get("hands", [])

    # ── STEP 1: NO HAND VISIBLE — idle / resting state ───────────────────────
    # No hand detected → reset stability and trigger memory.
    # Return an idle message; the robot does nothing.
    if not hands:
        LAST_TRIGGER_KEY = None
        STABILITY        = {}
        return {
            "label":      "No hand detected",
            "confidence": 0.0,
            "box_posteriors": {k: 1.0 / max(len(centers), 1) for k in centers},
            "debug_text": [
                "Show one or two hands to the webcam.",
                "Move toward the red or green target, or use a recognisable gesture.",
                f"Trigger fires when P(G|O) > {CONFIRM_THRESHOLD:.0%}.",
            ],
        }

    # ── STEP 2: NO TARGET BOXES — gesture-only fallback ──────────────────────
    # Simulator has no coloured boxes → full Bayesian spatial model can't run.
    # Fall back to pure gesture mapping, still with stability + cooldown checks.
    if not centers:
        hand     = hands[0]
        side     = resolved_hand_label(hand)
        gesture, g_conf = classify_gesture(hand)
        op_idx   = FOUR_OPERATION_MAP.get((side, gesture))
        operation = operation_by_index(frame, op_idx)

        trigger_key = f"{side}:{gesture}"
        now         = frame.get("timestamp_ms", 0)
        stable      = update_stability([trigger_key])
        s_frames    = stable.get(trigger_key, 1)

        should_trigger = (
            operation is not None
            and gesture in ACTIONABLE_GESTURE_NAMES
            and g_conf >= 0.82
            and s_frames >= REQUIRED_STABLE_FRAMES
            and (trigger_key != LAST_TRIGGER_KEY or now - LAST_TRIGGER_AT > COOLDOWN_MS)
        )
        if should_trigger:
            LAST_TRIGGER_KEY = trigger_key
            LAST_TRIGGER_AT  = now

        return {
            "label":      f"{side} hand (gesture-only): {gesture}",
            "confidence": g_conf,
            "trigger_operation_id":
                operation["id"] if should_trigger and operation else None,
            "cooldown_ms": COOLDOWN_MS,
            "debug_text": [
                "No Bayesian boxes found — using gesture mapping only.",
                f"Gesture: {gesture} | stable: {s_frames} frames",
                f"Mapped: {operation['name'] if operation else 'none'}",
            ],
        }

    # ── STEP 3: FULL BAYESIAN PIPELINE (one pass per visible hand) ───────────
    # Run _process_hand() for every hand in frame. Each hand independently
    # maintains its own posterior distribution. Results are collected below.
    results       = []
    stability_keys = []

    for hand in hands:
        result = _process_hand(frame, hand, centers, boxes)
        results.append(result)
        stability_keys.append(result["stability_key"])

    # ── STEP 4: UPDATE STABILITY COUNTERS ────────────────────────────────────
    # Increment per-hand consecutive-frame counters, then attach each
    # hand's current stable_frames count to its result dict.
    stability = update_stability(stability_keys)
    for r in results:
        r["stable_frames"] = stability.get(r["stability_key"], 1)

    # ── STEP 5: CHOOSE THE "ACTIVE" HAND ─────────────────────────────────────
    # When two hands are visible, pick the most intentional one using a
    # three-level priority (evaluated left-to-right as a tuple):
    #   1. Is the hand physically inside a target box?  (strongest signal)
    #   2. How many stable consecutive frames has it accumulated?
    #   3. How high is its posterior confidence?        (tiebreaker)
    active = max(
        results,
        key=lambda r: (
            1 if r["inside_green"] or r["inside_red"] else 0,
            r["stable_frames"],
            r["confidence"],
        ),
    )

    # ── STEP 6: TRIGGER DECISION — THE CONFIRMATION GATE ─────────────────────
    # The robot fires ONLY when ALL five conditions are true simultaneously:
    #   ✓ 1. A matching saved operation exists
    #   ✓ 2. Posterior confidence > 80%  (CONFIRM_THRESHOLD)
    #   ✓ 3. Same intent held for ≥ 2 consecutive frames
    #   ✓ 4. Hand is inside a box OR making an actionable gesture
    #   ✓ 5. ≥ 1400 ms have passed since the last trigger  (cooldown)
    trigger_key  = f"{active['side']}:{active['intent']}:{active['gesture']}"
    now          = frame.get("timestamp_ms", 0)
    in_box_flag  = active["inside_green"] or active["inside_red"]

    should_trigger = (
        active["operation"] is not None
        and active["confidence"] >= CONFIRM_THRESHOLD
        and active["stable_frames"] >= REQUIRED_STABLE_FRAMES
        and (in_box_flag or active["gesture"] in ACTIONABLE_GESTURE_NAMES)
        and (trigger_key != LAST_TRIGGER_KEY or now - LAST_TRIGGER_AT > COOLDOWN_MS)
    )

    if should_trigger:
        LAST_TRIGGER_KEY = trigger_key
        LAST_TRIGGER_AT  = now
    elif active["gesture"] == "hover" and not in_box_flag:
        LAST_TRIGGER_KEY = None     # reset when hand is idle

    # ── STEP 7: BUILD LIVE DEBUG OUTPUT ──────────────────────────────────────
    # Compose the text shown in the simulator's webcam overlay panel.
    # Shows posteriors, gesture, confidence, and trigger status each frame.
    debug_lines = []

    # Per-hand summary
    for r in results:
        post_str = " | ".join(
            f"P({k})={v:.2f}" for k, v in sorted(r["posterior"].items())
        )
        debug_lines.append(
            f"{r['side']}: {r['gesture']} → {r['intent']} | {post_str}"
        )

    # Active hand detail
    debug_lines.append(
        f"Active: {active['side']} | conf={active['confidence']:.2f} "
        f"| stable={active['stable_frames']}f"
    )

    # Operation being targeted
    if active["operation"]:
        debug_lines.append(f"Target op: {active['operation']['name']}")
    else:
        if active["operation_index"] is not None:
            debug_lines.append(
                f"Waiting: saved op #{active['operation_index']+1} not found"
            )
        else:
            debug_lines.append("No mapping for this intent/gesture combo")

    # Trigger status
    if should_trigger:
        debug_lines.append(
            f"★ TRIGGERED: {active['operation']['name']} "
            f"(P={active['confidence']:.2f} > {CONFIRM_THRESHOLD:.2f})"
        )
    else:
        needed = CONFIRM_THRESHOLD - active["confidence"]
        debug_lines.append(
            f"Needs Δ={needed:+.2f} more confidence to trigger" if needed > 0
            else "Cooldown or stability check active"
        )

    # ── STEP 8: AGGREGATE POSTERIORS FOR THE COLOURED BOX OVERLAY ────────────
    # When two hands are visible, use the HIGHER of the two posteriors per box
    # so the overlay reflects whichever hand is currently more active.
    agg_posteriors = {}
    for box_id in centers:
        vals = [r["posterior"].get(box_id, 0.0) for r in results]
        agg_posteriors[box_id] = max(vals)

    return {
        "label": (
            f"{active['side']}: {active['gesture']} → {active['intent']} "
            f"({active['confidence']:.0%})"
        ),
        "confidence":          active["confidence"],
        "trigger_operation_id": (
            active["operation"]["id"]
            if should_trigger and active["operation"]
            else None
        ),
        "trigger_operation_name": (
            active["operation"]["name"]
            if should_trigger and active["operation"]
            else None
        ),
        "cooldown_ms":  COOLDOWN_MS,
        "box_posteriors": agg_posteriors,
        "debug_text":   debug_lines[:8],   # simulator shows max ~8 lines
    }

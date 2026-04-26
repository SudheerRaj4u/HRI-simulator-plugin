"""
EE653 Human-Robot Interaction — Final Project Plugin
=====================================================
Author  : [Your Name]
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

import math

# ---------------------------------------------------------------------------
# Plugin metadata (shown in the simulator UI)
# ---------------------------------------------------------------------------
PLUGIN_META = {
    "name": "Bayesian Goal-Aware Handover",
    "description": (
        "Combines gesture recognition and Bayesian spatial intent inference "
        "to infer which robot handover operation the user wants. "
        "Triggers when P(G|O) > 0.80 using Bayesian Goal Recognition."
    ),
}

# ---------------------------------------------------------------------------
# Hyper-parameters — adjust these to tune the plugin behaviour
# ---------------------------------------------------------------------------

# Gaussian likelihood spread (σ). Smaller = tighter, more spatially precise.
SIGMA = 0.13

# Boost applied to the likelihood of whichever box the fingertip is inside.
INSIDE_BOX_BOOST = 12.0
# Penalty on the *other* box when the fingertip is clearly inside one box.
INSIDE_BOX_PENALTY = 0.07

# Gesture evidence weights — how much a detected gesture shifts the likelihood
# toward the associated goal (applied as a multiplicative boost).
GESTURE_BOOST = 2.5

# HMM transition parameters (Equation 2)
# STAY_PROB: how sticky intent is. Higher = smoother but slower to switch.
STAY_PROB = 0.82
SWITCH_PROB = 1.0 - STAY_PROB          # probability mass shared across others

# Confirmation threshold — robot only triggers when P(G*|O) exceeds this.
CONFIRM_THRESHOLD = 0.80

# Minimum consecutive stable frames before a trigger is allowed.
REQUIRED_STABLE_FRAMES = 2

# Cooldown in milliseconds after a trigger fires (prevents rapid re-fires).
COOLDOWN_MS = 1400

# Four-operation gesture map.
# Key  : (hand_side, gesture_name)
# Value: index into frame["saved_operations"]
# Students: edit this table to link different gestures to different operations.
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

# Gesture priority for tie-breaking when multiple hands are visible.
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
# Global mutable state (reset in setup())
# ---------------------------------------------------------------------------
# POSTERIOR[side] = {"red": float, "green": float}  — per-hand Bayes state
POSTERIOR   = {}
STABILITY   = {}               # maps state_key -> consecutive frame count
LAST_TRIGGER_KEY = None
LAST_TRIGGER_AT  = 0


# ===========================================================================
# Lifecycle: setup
# ===========================================================================
def setup(payload):
    """
    Called once when the plugin is loaded.
    Resets all global state and reports ready status to the simulator.
    """
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
# Helper utilities
# ===========================================================================

def normalize(prob_dict):
    """Ensure probabilities sum to 1.0 (Equation 1 denominator)."""
    total = sum(prob_dict.values()) or 1.0
    return {k: v / total for k, v in prob_dict.items()}


def bayes_update(prior, likelihood):
    """
    Apply one step of Bayes rule (Equation 1).

          posterior[G] = likelihood[G] * prior[G]   (unnormalized)
    Then normalize so all posteriors sum to 1.
    """
    unnormalized = {g: likelihood[g] * prior[g] for g in prior}
    return normalize(unnormalized)


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


def inside_box(point, box):
    """Return True if the fingertip/palm point is inside the given box."""
    return (
        box["x"] <= point["x"] <= box["x"] + box["width"]
        and box["y"] <= point["y"] <= box["y"] + box["height"]
    )


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


# ---------------------------------------------------------------------------
# Gesture classification (mirrors gesture_operation_mapping.py logic)
# ---------------------------------------------------------------------------

def _finger_states(hand):
    return hand.get("fingerStates") or {}


def _count_no_thumb(hand):
    rv = hand.get("fingerCountNoThumb")
    if rv is not None:
        return int(rv)
    fs = _finger_states(hand)
    return sum(
        1 for k in ("indexExtended", "middleExtended", "ringExtended", "pinkyExtended")
        if fs.get(k)
    )


def _count_with_thumb(hand):
    rv = hand.get("fingerCountWithThumb")
    if rv is not None:
        return int(rv)
    fs = _finger_states(hand)
    return _count_no_thumb(hand) + (1 if fs.get("thumbExtended") else 0)


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


def resolved_hand_label(hand):
    raw = hand.get("handedness") or hand.get("viewerSide") or "Unknown"
    # Normalise to title-case so "left", "LEFT", "Left" all match the map
    return raw.strip().title()


# ---------------------------------------------------------------------------
# Operation lookup
# ---------------------------------------------------------------------------

def operation_by_index(frame, index):
    """Safely retrieve a saved operation by list index. Returns None if missing."""
    saved = frame.get("saved_operations", [])
    if index is None or index < 0 or index >= len(saved):
        return None
    return saved[index]


def mapped_operation_for(frame, hand_label, gesture, intent):
    """
    Map (hand_label, gesture, intent) to a saved operation.

    Priority:
      1. Four-operation gesture map (if ≥ 4 saved operations exist).
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


# ---------------------------------------------------------------------------
# Stability tracking
# ---------------------------------------------------------------------------

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
# Core per-frame logic
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

    # --- 1. Initialise prior if this hand is newly visible ----------------
    if side not in POSTERIOR:
        hypotheses      = list(centers.keys())
        n               = len(hypotheses) or 1
        POSTERIOR[side] = {h: 1.0 / n for h in hypotheses}

    # --- 2. HMM prediction step (Equation 2) ------------------------------
    prior = hmm_predict(POSTERIOR[side])

    # --- 3. Compute spatial likelihood P(O | G) per hypothesis ------------
    likelihood = {}
    for box_id, center in centers.items():
        l_val = gaussian_likelihood(point, center)

        # Inside-box bonus/penalty
        in_box  = inside_box(point, boxes[box_id])
        out_ids = [b for b in boxes if b != box_id]
        out_box = any(inside_box(point, boxes[b]) for b in out_ids)

        if in_box and not out_box:
            l_val *= INSIDE_BOX_BOOST
        elif out_box and not in_box:
            l_val *= INSIDE_BOX_PENALTY

        # Gesture-based evidence boost:
        # If the active gesture maps to an operation index that corresponds
        # to this box position, boost its likelihood.
        op_idx = FOUR_OPERATION_MAP.get((side, gesture))
        if op_idx is not None:
            saved = frame.get("saved_operations", [])
            # green = even indices (0, 2), red = odd indices (1, 3)
            if (box_id == "green" and op_idx % 2 == 0) or \
               (box_id == "red"   and op_idx % 2 == 1):
                l_val *= GESTURE_BOOST

        likelihood[box_id] = max(l_val, 1e-9)   # avoid zero likelihood

    # --- 4. Bayes update — compute posterior (Equation 1) -----------------
    posterior       = bayes_update(prior, likelihood)
    POSTERIOR[side] = posterior

    # --- 5. Determine winning intent hypothesis ----------------------------
    intent     = max(posterior, key=posterior.get)
    confidence = posterior[intent]

    inside_red   = inside_box(point, boxes.get("red",   {})) if "red"   in boxes else False
    inside_green = inside_box(point, boxes.get("green", {})) if "green" in boxes else False

    # --- 6. Map to a saved operation --------------------------------------
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

def process_frame(frame):
    """
    Main entry point called by the simulator on every webcam frame.

    Pipeline:
        Perception → Inference (Bayes + HMM) → Confirmation (>0.80) → Trigger
    """
    global LAST_TRIGGER_AT, LAST_TRIGGER_KEY, POSTERIOR, STABILITY

    # ------------------------------------------------------------------
    # Step 0: Read target boxes
    # ------------------------------------------------------------------
    centers, boxes = box_centers(frame)
    hands          = frame.get("hands", [])

    # ------------------------------------------------------------------
    # Step 1: No hand — reset and return safe state
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Step 2: No boxes — fall back to gesture-only mode
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Step 3: Full Bayesian pipeline — one entry per visible hand
    # ------------------------------------------------------------------
    results       = []
    stability_keys = []

    for hand in hands:
        result = _process_hand(frame, hand, centers, boxes)
        results.append(result)
        stability_keys.append(result["stability_key"])

    # ------------------------------------------------------------------
    # Step 4: Stability update
    # ------------------------------------------------------------------
    stability = update_stability(stability_keys)
    for r in results:
        r["stable_frames"] = stability.get(r["stability_key"], 1)

    # ------------------------------------------------------------------
    # Step 5: Select the most active hand
    # Priority: (inside a box) > (stable frames) > (posterior confidence)
    # ------------------------------------------------------------------
    active = max(
        results,
        key=lambda r: (
            1 if r["inside_green"] or r["inside_red"] else 0,
            r["stable_frames"],
            r["confidence"],
        ),
    )

    # ------------------------------------------------------------------
    # Step 6: Trigger decision — confirm when P(G*|O) > CONFIRM_THRESHOLD
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Step 7: Build rich debug output (shown live in the webcam panel)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Step 8: Aggregate box posteriors across all hands for the overlay
    # ------------------------------------------------------------------
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

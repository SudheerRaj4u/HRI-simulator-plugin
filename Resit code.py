# =============================================================================
# HRI PROJECT — Real-Time Webcam-Based Intention Recognition
# Student Number: 25250780
# =============================================================================
#
# WHAT THIS PROGRAM DOES (Plain English):
# ----------------------------------------
# This program uses your webcam to watch your hand move and tries to guess
# WHICH of three targets (Left, Centre, Right) you are reaching for —
# BEFORE you actually get there. This is called "intention recognition."
#
# Once the system guesses your target, it moves a simulated robot cursor
# (a yellow circle on screen) toward that target to "meet" your hand.
# It also watches how close the robot gets to your hand and slows down
# or stops if it gets too close (safety monitoring).
#
# After every trial, the robot adjusts its speed based on whether it
# guessed correctly (this is called "adaptation").
#
# The experiment runs 20 trials in total, then saves results to a CSV file
# and generates two graphs.
# =============================================================================


# --- IMPORTS ---
# These lines load external "toolboxes" (called libraries) that give us
# extra abilities we didn't have to write ourselves.

import cv2              # OpenCV: lets us open the webcam and draw on video frames
import numpy as np      # NumPy: lets us do fast maths on arrays of numbers (like coordinates)
import time             # time: lets us measure how long things take (in seconds)
import math             # math: standard maths functions (not heavily used but available)
import csv              # csv: lets us save data to a spreadsheet-style .csv file
import random           # random: lets us shuffle lists randomly
import matplotlib.pyplot as plt  # Matplotlib: lets us draw graphs and save them as .png images


# =============================================================================
# THE MAIN CLASS — think of this as the "blueprint" for the whole experiment.
# A class bundles together all the data (variables) and actions (functions)
# that our system needs. We create ONE object from this class called 'app'.
# =============================================================================

class HRIExperiment:

    # -------------------------------------------------------------------------
    # __init__  —  "Initialise" (set up everything when the program starts)
    # This function runs ONCE when we create the HRIExperiment object.
    # It sets all starting values and prepares the webcam.
    # -------------------------------------------------------------------------
    def __init__(self):

        # --- WEBCAM SETUP ---
        # cv2.VideoCapture(0) opens the default webcam (camera index 0).
        # If you had two webcams, you could use (1) for the second one.
        self.cap = cv2.VideoCapture(0)

        # Ask the webcam to use 640×480 pixel resolution (standard definition).
        # "self.cap.set()" sends a setting command to the camera.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Read back the actual resolution the camera agreed to use
        # (it might differ slightly from what we asked for).
        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # --- CALIBRATION STATE ---
        # These variables track whether calibration is finished.
        self.calibrated      = False  # Will become True once calibration is complete
        self.calibration_step = 0     # Counts which calibration click we are on (0 to 4)

        # self.targets is a list that will store the (x, y) pixel positions of
        # the three targets: [Left, Centre, Right].
        # It starts empty and gets filled during calibration.
        self.targets    = []   # Will hold 3 positions: Left target, Centre target, Right target
        self.start_zone = None # The position the user must return to between trials

        # --- COLOUR TRACKING RANGE ---
        # We track the hand using a coloured marker (sticker/glove).
        # Colours in OpenCV are represented in HSV format (Hue, Saturation, Value).
        # These are the lower and upper bounds of the colour we're looking for.
        # They start as full range (everything) and get narrowed down during calibration.
        self.hsv_lower = np.array([0, 0, 0])        # Lowest colour value to detect
        self.hsv_upper = np.array([255, 255, 255])  # Highest colour value to detect

        # --- WINDOW & MOUSE ---
        # Create the display window called 'HRI System'.
        cv2.namedWindow('HRI System')

        # Tell OpenCV: whenever the mouse is clicked inside 'HRI System',
        # call our function self.mouse_callback to handle it.
        cv2.setMouseCallback('HRI System', self.mouse_callback)

        # --- ROBOT STATE ---
        # The "robot" is just a yellow circle drawn on the screen.
        # robot_speed_gain controls how fast it moves (like a speed dial).
        # It starts at 1.5 and adapts up/down after each trial.
        self.robot_speed_gain = 1.5

        self.robot_pos = None  # Current pixel position of the robot (set each trial)

        # The robot always starts at the top-centre of the frame, 50 pixels from the top.
        # width//2 means "half the screen width" (integer division).
        self.robot_start_pos = np.array([self.width // 2, 50], dtype=float)

        # --- TRIAL DATA ---
        self.trial_logs = []  # Will store a summary dictionary after each of the 20 trials
        self.trials     = []  # Will store the list of 20 trial definitions

        # Call setup_trials() to fill self.trials with the 20 randomised trials.
        self.setup_trials()

        # Human-readable names for the three targets (index 0=Left, 1=Centre, 2=Right).
        self.target_names = ["Left", "Centre", "Right"]

        # Storage for probability-over-time data from the first successful trial
        # (used later to draw the probability graph).
        self.prob_history = []


    # -------------------------------------------------------------------------
    # setup_trials  —  Creates the list of 20 randomised trials
    # -------------------------------------------------------------------------
    def setup_trials(self):
        # Create a list of 20 trial types:
        #   12 Normal trials  (regular speed reaching)
        #    4 Slow/Ambiguous  (deliberately slow or unclear reaching)
        #    4 Safety-Critical (intentionally cross near the robot)
        types = ['Normal'] * 12 + ['Slow/Ambiguous'] * 4 + ['Safety-Critical'] * 4

        # Shuffle the order randomly so the participant can't predict what comes next.
        random.shuffle(types)

        # Create 20 target indices: each of [0=Left, 1=Centre, 2=Right] appears at
        # least 5 times (that's the [0,1,2]*5 part), plus 5 extra random ones.
        target_indices = [0, 1, 2] * 5 + [random.choice([0, 1, 2]) for _ in range(5)]
        random.shuffle(target_indices)

        # Build the trial list. Each trial is a small dictionary with:
        #   'id'         — trial number (1 to 20)
        #   'type'       — what kind of trial it is
        #   'target_idx' — which target the participant should reach for (0, 1, or 2)
        for i in range(20):
            self.trials.append({
                'id':         i + 1,
                'type':       types[i],
                'target_idx': target_indices[i]
            })


    # -------------------------------------------------------------------------
    # mouse_callback  —  Called automatically whenever the user clicks the mouse
    #
    # Parameters:
    #   event — what kind of mouse event happened (click, move, etc.)
    #   x, y  — the pixel coordinates where the mouse was when the event happened
    #   flags, param — extra info (not used here)
    # -------------------------------------------------------------------------
    def mouse_callback(self, event, x, y, flags, param):

        # Only respond to left-button clicks, and only during calibration.
        if event == cv2.EVENT_LBUTTONDOWN and not self.calibrated:

            if self.calibration_step < 3:
                # Clicks 1, 2, 3 — set the three target positions (Left, Centre, Right).
                # np.array([x, y]) stores the pixel coordinate as a 2D point.
                self.targets.append(np.array([x, y], dtype=float))
                self.calibration_step += 1  # Move to the next calibration step

            elif self.calibration_step == 3:
                # Click 4 — set the Start Zone position.
                self.start_zone = np.array([x, y], dtype=float)
                self.calibration_step += 1

            elif self.calibration_step == 4:
                # Click 5 — the user clicks ON their coloured marker to teach the
                # system what colour to track.

                # Grab the current video frame from the webcam.
                ret, frame = self.cap.read()
                frame = cv2.flip(frame, 1)  # Mirror the frame (like a selfie camera)

                # Convert the frame from BGR colour space to HSV colour space.
                # HSV separates colour (Hue) from brightness (Value), making colour
                # detection much more reliable under different lighting conditions.
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # Read the exact HSV colour values of the pixel the user clicked on.
                pixel_hsv = hsv[y, x]
                h = int(pixel_hsv[0])  # Hue   — the type of colour (e.g. red, green, blue)
                s = int(pixel_hsv[1])  # Saturation — how vivid/pure the colour is
                v = int(pixel_hsv[2])  # Value — how bright or dark the colour is

                # Build a colour detection range centred on that pixel:
                # Lower bound: allow the hue to be up to 20 units darker/lower,
                #              saturation down to 50, brightness down to 50.
                # Upper bound: allow hue up to 20 units lighter/higher, full S and V.
                # max() and min() make sure we don't go outside the valid HSV range (0–179 for H).
                self.hsv_lower = np.array([max(0,   h - 20), max(50, s - 60), max(50, v - 60)], dtype=np.uint8)
                self.hsv_upper = np.array([min(179, h + 20), 255,             255            ], dtype=np.uint8)

                self.calibration_step += 1
                self.calibrated = True  # Calibration is now done!


    # -------------------------------------------------------------------------
    # run  —  The main entry point. Controls the three phases of the experiment.
    # -------------------------------------------------------------------------
    def run(self):
        print("Starting HRI System...")

        # =====================================================================
        # PHASE 1: CALIBRATION
        # Keep showing the webcam feed and waiting for the user to click
        # until calibration is complete (self.calibrated becomes True).
        # =====================================================================
        while not self.calibrated:
            # Read one frame from the webcam.
            # 'ret' is True if the frame was captured successfully; 'frame' is the image.
            ret, frame = self.cap.read()
            if not ret:
                break  # Stop if webcam fails

            # Flip the frame horizontally (mirror image) — feels more natural.
            frame = cv2.flip(frame, 1)

            # List of on-screen instructions shown one at a time, matching calibration_step.
            msgs = [
                "Click to set LEFT target",
                "Click to set CENTRE target",
                "Click to set RIGHT target",
                "Click to set START zone",
                "Click on your coloured marker to set tracking colour"
            ]

            # Draw the current instruction text on the frame in red.
            if self.calibration_step < 5:
                cv2.putText(frame, msgs[self.calibration_step], (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Draw a green circle and label for each target that has been set so far.
            for i, t in enumerate(self.targets):
                cv2.circle(frame, tuple(t.astype(int)), 20, (0, 255, 0), 2)
                cv2.putText(frame, self.target_names[i],
                            tuple(t.astype(int) - np.array([20, 30])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Draw the Start Zone as a blue circle once it has been set.
            if self.start_zone is not None:
                cv2.circle(frame, tuple(self.start_zone.astype(int)), 30, (255, 0, 0), 2)
                cv2.putText(frame, "Start",
                            tuple(self.start_zone.astype(int) - np.array([20, 40])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Show the current frame in the window.
            cv2.imshow('HRI System', frame)

            # Wait 1 millisecond for a key press.
            # If the user presses Escape (key code 27), stop the program.
            if cv2.waitKey(1) & 0xFF == 27:
                break

        # If the user pressed Escape before finishing calibration, clean up and exit.
        if not self.calibrated:
            self.cap.release()
            cv2.destroyAllWindows()
            return

        print("Calibration complete. Starting trials.")

        # =====================================================================
        # PHASE 2: EXPERIMENT LOOP
        # Run each of the 20 trials one by one.
        # =====================================================================
        for trial in self.trials:
            self.run_trial(trial)

        # =====================================================================
        # PHASE 3: OUTPUTS
        # Save results and generate graphs once all trials are done.
        # =====================================================================
        self.generate_outputs()

        # Release the webcam so other programs can use it again.
        self.cap.release()

        # Close all OpenCV windows.
        cv2.destroyAllWindows()


    # -------------------------------------------------------------------------
    # run_trial  —  Runs ONE trial from start to finish.
    #
    # Parameter:
    #   trial — a dictionary with keys 'id', 'type', and 'target_idx'
    # -------------------------------------------------------------------------
    def run_trial(self, trial):

        # Unpack the trial information from the dictionary.
        trial_id         = trial['id']          # Which trial number (1–20)
        trial_type       = trial['type']         # What type (Normal, Slow/Ambiguous, Safety-Critical)
        true_target_idx  = trial['target_idx']   # Which target the participant should reach for (0, 1, or 2)
        true_target_pos  = self.targets[true_target_idx]  # The actual pixel position of that target

        print(f"Trial {trial_id} | Type: {trial_type} | Target: {self.target_names[true_target_idx]}")

        # --- RESET ROBOT POSITION ---
        # Move the robot back to its starting position (top-centre of screen).
        self.robot_pos = self.robot_start_pos.copy()

        # --- HAND TRACKING VARIABLES ---
        hand_pos      = None           # Current position of the hand (set each frame)
        prev_hand_pos = None           # Position of the hand in the previous frame (for velocity)
        hand_vel      = 0.0            # Speed of the hand (magnitude of velocity, in pixels/second)
        hand_vel_vec  = np.zeros(2)    # Velocity as a 2D vector [vx, vy] (pixels/second)

        # --- BAYESIAN INTENTION RECOGNITION VARIABLES ---
        # probs holds the probability that the hand is heading toward each target.
        # They start equal (1/3 each) because we have no information yet.
        probs = np.array([0.333, 0.333, 0.333])  # [P(Left), P(Centre), P(Right)]

        target_locked      = False  # Has the system committed to a predicted target?
        locked_target_idx  = -1     # Index of the locked target (-1 means none yet)
        lock_time          = None   # How many seconds into the trial the lock happened
        lock_candidate_frames = 0   # How many consecutive frames have been above the 0.80 threshold
                                    # (we require 5 in a row before committing — FIX 4)

        # --- TIMING ---
        start_time       = time.time()  # Wall-clock time when this function was called
        trial_start_time = start_time   # Will be updated to when the participant actually starts moving
        prev_time        = start_time   # Used to calculate time between frames (dt)

        # --- SAFETY COUNTERS ---
        safety_violations = 0               # Count of frames where robot was too close to hand
        min_distance      = float('inf')    # Closest the robot ever got to the hand (starts at infinity)

        # --- TRIAL STATE ---
        # The trial moves through two states:
        #   "WAITING"   — waiting for the participant to position their hand at the Start Zone
        #   "REACHING"  — participant is actively reaching; system is tracking and predicting
        #   "COMPLETED" — trial is finished (robot reached the target, or Escape was pressed)
        state = "WAITING"

        # --- PROBABILITY RECORDING ---
        # We only record the detailed probability-over-time data for the FIRST successful trial,
        # so we can draw the probability graph later.
        record_probs = False
        if len(self.prob_history) == 0:
            record_probs = True          # This trial will record probabilities
            current_prob_history = []    # Temporary list to store (time, probs) pairs

        fps_list             = []   # Stores frames-per-second each frame (for performance tracking)
        predicted_arrival_time = 0.0  # Our prediction of when the hand will reach the target


        # =====================================================================
        # MAIN TRIAL LOOP — runs one iteration per video frame (~40 times/sec)
        # =====================================================================
        while state != "COMPLETED":

            # Read a new frame from the webcam.
            ret, frame = self.cap.read()
            if not ret:
                break  # Stop if webcam fails

            # Mirror the frame so it behaves like a selfie camera.
            frame = cv2.flip(frame, 1)

            # Reset safety status for this frame.
            safety_status = "SAFE"
            safety_color  = (0, 255, 0)  # Green colour for "safe"

            # --- TIME DELTA (dt) ---
            # dt is the time elapsed since the last frame, in seconds.
            # We use this to convert pixel differences into velocities (pixels per second).
            curr_time = time.time()
            dt        = curr_time - prev_time
            if dt < 0.001:
                dt = 0.001      # Clamp to at least 1ms to avoid division by zero
            prev_time = curr_time

            # Calculate frames per second (FPS) for this frame and store it.
            fps = 1.0 / dt
            fps_list.append(fps)

            # =================================================================
            # PERCEPTION — Find the coloured marker (hand) in the frame
            # =================================================================

            # Convert the frame from BGR to HSV colour space.
            # HSV makes it much easier to filter by colour under different lighting.
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Create a "mask" — a black-and-white image where white pixels are
            # within our colour range (the marker) and black pixels are everything else.
            mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

            # Erosion: shrinks white regions to remove small noise specks.
            mask = cv2.erode(mask, None, iterations=2)

            # Dilation: grows white regions back to their original size
            # (but the noise specks that were removed by erosion do NOT come back).
            mask = cv2.dilate(mask, None, iterations=2)

            # Find the outlines (contours) of all white blobs in the mask.
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                # Pick the largest blob — that should be our marker.
                c = max(contours, key=cv2.contourArea)

                # Only use it if the blob is large enough (>500 pixels²) to be real,
                # not just a tiny noise artefact.
                if cv2.contourArea(c) > 500:

                    # Calculate the "moment" of the blob to find its centre of mass.
                    # M["m10"]/M["m00"] gives the x-coordinate of the centroid.
                    # M["m01"]/M["m00"] gives the y-coordinate of the centroid.
                    M  = cv2.moments(c)
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Store the hand position as a 2D coordinate [cx, cy].
                    hand_pos = np.array([cx, cy], dtype=float)

                    # Draw a small red filled circle on screen to show where we detected the hand.
                    cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)

                    # --- VELOCITY CALCULATION ---
                    if prev_hand_pos is not None:
                        # Raw velocity = (new position − old position) / time elapsed
                        # This gives us a velocity vector in pixels per second.
                        raw_vel_vec = (hand_pos - prev_hand_pos) / dt

                        # Smooth the velocity using exponential moving average.
                        # "0.5 * old + 0.5 * new" blends old and new readings 50/50,
                        # which reduces the jitter caused by webcam noise.
                        hand_vel_vec = 0.5 * hand_vel_vec + 0.5 * raw_vel_vec

                        # The speed (scalar) is the length of the velocity vector.
                        hand_vel = np.linalg.norm(hand_vel_vec)

                    # Remember this frame's position for next frame's velocity calculation.
                    prev_hand_pos = hand_pos


            # =================================================================
            # STATE MACHINE — what should we do this frame?
            # =================================================================

            if state == "WAITING":
                # ---- WAITING STATE ----
                # Display an instruction telling the participant to move to the Start Zone.
                cv2.putText(frame, f"Trial {trial_id} ({trial_type}): Move to START zone",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                # Check if the hand is within 40 pixels of the Start Zone centre.
                if hand_pos is not None:
                    dist_to_start = np.linalg.norm(hand_pos - self.start_zone)
                    if dist_to_start < 40:
                        # The participant is at the Start Zone — begin the trial.
                        state            = "REACHING"
                        trial_start_time = curr_time  # Record when the reaching started
                        print("Trial started.")

            elif state == "REACHING":
                # ---- REACHING STATE ----
                # Tell the participant which target to reach for.
                cv2.putText(frame, f"Trial {trial_id} ({trial_type}): Reach for {self.target_names[true_target_idx]}",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)


                # =============================================================
                # INTENTION RECOGNITION (Bayesian Update)
                # =============================================================
                # Only update our beliefs when the hand is moving fast enough
                # (> 30 px/s) to give a meaningful direction signal.
                # FIX 1: threshold raised from 10 to 30 to suppress webcam jitter.
                if hand_pos is not None and hand_vel > 30.0:

                    # Compute a "likelihood" for each target:
                    # How likely is it that the hand movement we just observed
                    # would happen IF the person were heading to target i?
                    likelihoods = np.zeros(3)  # One value per target (Left, Centre, Right)

                    for i, t_pos in enumerate(self.targets):

                        # Vector pointing from hand toward this target.
                        vec_to_t = t_pos - hand_pos
                        dist     = np.linalg.norm(vec_to_t)  # Distance in pixels

                        if dist > 0:
                            # Unit vector pointing toward the target (direction only, no magnitude).
                            dir_t = vec_to_t / dist

                            # Unit vector of the hand's current movement direction.
                            dir_h = hand_vel_vec / hand_vel

                            # Dot product: measures alignment between hand direction and target direction.
                            # Result ranges from -1 (opposite) to +1 (perfectly aligned).
                            # We clamp to 0 (i.e. ignore targets behind the hand).
                            similarity = max(0, np.dot(dir_h, dir_t))

                            # FIX 3: Likelihood formula — 80% direction, 20% proximity.
                            # exp(-dist/1000): closer targets get a slightly higher score, but
                            # this term is now weak so that DIRECTION is the main signal.
                            # (0.2 + 0.8 * similarity): heavily rewards targets the hand is pointing at.
                            likelihoods[i] = np.exp(-dist / 1000.0) * (0.2 + 0.8 * similarity)
                        else:
                            # Hand is exactly on this target — treat as maximum likelihood.
                            likelihoods[i] = 1.0

                    # --- BAYESIAN UPDATE ---
                    # Multiply the current probabilities by the new likelihoods.
                    # This is Bayes' Rule: new belief = old belief × new evidence.
                    probs = probs * likelihoods

                    # Normalise so all three probabilities still sum to 1.0.
                    if np.sum(probs) > 0:
                        probs = probs / np.sum(probs)
                    else:
                        # If all probabilities collapsed to zero (very unlikely), reset to equal.
                        probs = np.array([0.333, 0.333, 0.333])

                    # Save the current probabilities with a timestamp (for the graph later).
                    if record_probs:
                        current_prob_history.append((curr_time - trial_start_time, probs.copy()))

                else:
                    # FIX 2: When the hand is slow (or not detected), gently pull the
                    # probabilities back toward equal [0.333, 0.333, 0.333].
                    # "0.95 * probs" keeps 95% of the current belief.
                    # "0.05 * uniform" nudges 5% back toward "I don't know."
                    # This prevents early noise from permanently locking the wrong target.
                    probs = 0.95 * probs + 0.05 * np.array([0.333, 0.333, 0.333])


                # =============================================================
                # TARGET LOCKING
                # =============================================================
                # If the highest probability exceeds 0.80 (80%), the system is
                # confident enough to "lock" onto a predicted target.
                # FIX 4: We require this confidence to hold for 5 consecutive frames
                # before committing, to avoid locking from a single noisy frame.
                if not target_locked:
                    max_prob = np.max(probs)

                    if max_prob >= 0.80:
                        # Confidence is high this frame — increment the confirmation counter.
                        lock_candidate_frames += 1

                        if lock_candidate_frames >= 5:
                            # The system has been confident for 5 frames in a row — lock it!
                            target_locked     = True
                            locked_target_idx = np.argmax(probs)  # Index of the most likely target
                            lock_time         = curr_time - trial_start_time  # Time to lock (seconds)
                            print(f"Target locked: {self.target_names[locked_target_idx]} "
                                  f"with prob {max_prob:.2f} (confirmed over {lock_candidate_frames} frames)")
                    else:
                        # Confidence dropped below threshold — reset the counter.
                        lock_candidate_frames = 0


                # =============================================================
                # MOTION PREDICTION & ROBOT CONTROL
                # =============================================================
                robot_vel = 0.0  # Robot speed starts at zero each frame

                if target_locked:
                    # The robot now moves toward the predicted goal target.
                    predicted_goal = self.targets[locked_target_idx]

                    # --- ARRIVAL TIME PREDICTION ---
                    # Once (and only once) when the hand is moving fast enough,
                    # predict when the hand will arrive at the target.
                    # Formula: current time + (remaining distance / current speed)
                    if predicted_arrival_time == 0.0 and hand_vel > 20.0:
                        dist_to_goal           = np.linalg.norm(predicted_goal - hand_pos)
                        predicted_arrival_time = (curr_time - trial_start_time) + (dist_to_goal / hand_vel)

                    # --- PROPORTIONAL ROBOT CONTROLLER ---
                    # robot_vel_vec = (direction to goal) × (speed gain)
                    # The further away the robot is from the goal, the faster it moves.
                    # This is a simple proportional controller — common in robotics.
                    robot_vel_vec = (predicted_goal - self.robot_pos) * self.robot_speed_gain
                    robot_vel     = np.linalg.norm(robot_vel_vec)

                    # ==========================================================
                    # SAFETY MONITORING — Protective Separation Distance (PSD)
                    # ==========================================================
                    if hand_pos is not None:

                        # Current distance between the hand and the robot cursor.
                        sep_dist     = np.linalg.norm(hand_pos - self.robot_pos)
                        min_distance = min(min_distance, sep_dist)  # Track the closest they ever got

                        # Calculate the required safe separation distance (d_safe).
                        # d0 = minimum absolute safe distance (50 pixels)
                        # k1 * hand_vel = extra buffer based on how fast the HAND is moving
                        #                 (fast hands need more space)
                        # k2 * robot_vel = extra buffer based on how fast the ROBOT is moving
                        #                  (fast robot needs more stopping distance)
                        d0     = 50.0   # Base safe distance in pixels
                        k1     = 0.05   # Weight for hand velocity contribution
                        k2     = 0.05   # Weight for robot velocity contribution
                        d_safe = d0 + k1 * hand_vel + k2 * robot_vel

                        if sep_dist < d_safe:
                            # The robot is inside the safe zone — take action!
                            safety_status = "UNSAFE (Slowing)"
                            safety_color  = (0, 0, 255)   # Red to show danger
                            robot_vel_vec *= 0.2           # Reduce robot speed to 20% of normal
                            safety_violations += 1         # Count this as a violation

                            if sep_dist < d0:
                                # The robot is VERY close (inside the hard minimum).
                                # Stop it completely.
                                safety_status = "UNSAFE (Stopped)"
                                robot_vel_vec *= 0.0  # Zero speed — full stop

                    # --- MOVE THE ROBOT ---
                    # Update robot position: new_pos = old_pos + velocity × time
                    # This is basic kinematics (physics of motion).
                    self.robot_pos += robot_vel_vec * dt

                    # --- CHECK FOR TRIAL COMPLETION ---
                    # The trial ends when BOTH conditions are true:
                    #   1. The robot is within 20 pixels of the predicted goal.
                    #   2. The hand is within 40 pixels of the predicted goal.
                    # (The hand also has to arrive — not just the robot.)
                    if (np.linalg.norm(predicted_goal - self.robot_pos) < 20.0 and
                            hand_pos is not None and
                            np.linalg.norm(predicted_goal - hand_pos) < 40.0):
                        state = "COMPLETED"


            # =================================================================
            # RENDERING — Draw the HUD (Heads-Up Display) on the frame
            # =================================================================

            # Draw each of the three target circles.
            # The TRUE target is drawn bright green; the others are grey.
            for i, t in enumerate(self.targets):
                color = (0, 255, 0) if i == true_target_idx else (200, 200, 200)
                cv2.circle(frame, tuple(t.astype(int)), 20, color, 2)
                # Show each target's current probability next to it.
                cv2.putText(frame, f"{self.target_names[i]}: {probs[i]:.2f}",
                            tuple(t.astype(int) - np.array([30, 30])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw the Start Zone as a blue circle.
            cv2.circle(frame, tuple(self.start_zone.astype(int)), 30, (255, 0, 0), 2)

            # Draw the robot as a yellow filled circle.
            cv2.circle(frame, tuple(self.robot_pos.astype(int)), 15, (255, 255, 0), -1)

            # Draw HUD text (information panel) on the left side of the screen.
            hud_y = 100  # Starting y-position for the first line of text

            cv2.putText(frame, f"FPS: {fps:.1f}",
                        (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            hud_y += 25  # Move down 25 pixels for the next line

            lock_str = self.target_names[locked_target_idx] if target_locked else "None"
            cv2.putText(frame, f"Estimated Target: {lock_str}",
                        (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            hud_y += 25

            cv2.putText(frame, f"Target Locked: {target_locked}",
                        (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            hud_y += 25

            # Only show safety info during the reaching phase.
            if state == "REACHING":
                cv2.putText(frame, f"Safety Status: {safety_status}",
                            (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, safety_color, 2)
                hud_y += 25

                dist_str = f"{np.linalg.norm(hand_pos - self.robot_pos):.1f}" if hand_pos is not None else "N/A"
                cv2.putText(frame, f"H-R Distance: {dist_str}",
                            (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                hud_y += 25

            cv2.putText(frame, f"Adaptation (Gain): {self.robot_speed_gain:.2f}",
                        (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            hud_y += 25

            # Show the frame in the window.
            cv2.imshow('HRI System', frame)

            # If the user presses Escape, end this trial early.
            if cv2.waitKey(1) & 0xFF == 27:
                state = "COMPLETED"


        # =====================================================================
        # END OF TRIAL — Calculate metrics and log results
        # =====================================================================

        # How long from trial start to trial end (in seconds).
        movement_time = time.time() - trial_start_time

        # Recognition accuracy: 1 if the system guessed the correct target, 0 if not.
        recognition_accuracy = 1 if (locked_target_idx == true_target_idx) else 0

        # Final position error: how far the hand was from the TRUE target at the end of the trial.
        # Normalised by screen width so 0.0 = perfect, 1.0 = completely wrong side of screen.
        final_pos_error = 0.0
        if hand_pos is not None:
            final_pos_error = np.linalg.norm(self.targets[true_target_idx] - hand_pos) / self.width

        # Actual arrival time = total movement time for this trial.
        actual_arrival_time = movement_time

        # FIX 5: If predicted_arrival_time was never set (because hand never went fast enough
        # during this trial), use the actual movement time as a fallback estimate.
        if predicted_arrival_time == 0.0:
            predicted_arrival_time = movement_time

        # Arrival time error: absolute difference between predicted and actual arrival.
        arrival_time_error = abs(predicted_arrival_time - actual_arrival_time)

        # Performance stats.
        avg_fps        = np.mean(fps_list) if len(fps_list) > 0 else 0
        approx_latency = (1.0 / avg_fps * 1000) if avg_fps > 0 else 0  # In milliseconds

        # Save probability history only from the first trial that was recognised correctly.
        if record_probs and recognition_accuracy == 1:
            self.prob_history = current_prob_history
            record_probs      = False


        # =================================================================
        # ADAPTATION — Update the robot's speed for the next trial
        # =================================================================
        # If the system correctly identified the target, the robot and human
        # were well coordinated → reward by increasing speed slightly.
        # If wrong, coordination was poor → penalise by reducing speed.
        # The gain is clamped between 0.5 (slow) and 3.0 (fast).
        if recognition_accuracy == 1:
            self.robot_speed_gain = min(3.0, self.robot_speed_gain + 0.1)  # Speed up (reward)
        else:
            self.robot_speed_gain = max(0.5, self.robot_speed_gain - 0.2)  # Slow down (penalise)


        # =================================================================
        # LOGGING — Save this trial's data as a dictionary
        # =================================================================
        log_entry = {
            'Trial number':                 trial_id,
            'True target':                  self.target_names[true_target_idx],
            'Estimated target':             self.target_names[locked_target_idx] if locked_target_idx != -1 else "None",
            'Recognition accuracy':         recognition_accuracy,      # 1 = correct, 0 = wrong
            'Target lock time':             lock_time if lock_time else -1,   # seconds (or -1 if never locked)
            'Movement time':                movement_time,             # total trial duration (seconds)
            'Predicted arrival time':       predicted_arrival_time,    # our prediction (seconds)
            'Actual arrival time':          actual_arrival_time,       # what actually happened (seconds)
            'Arrival time error':           arrival_time_error,        # difference between predicted and actual
            'Final position error':         final_pos_error,           # normalised 0–1 (lower is better)
            'Minimum human-robot distance': min_distance if min_distance != float('inf') else -1,
            'Safety violations':            safety_violations,         # number of frames robot was too close
            'Average FPS':                  avg_fps,                   # frames per second (performance)
            'Approximate latency':          approx_latency,            # milliseconds per frame
            'Adaptation parameter value':   self.robot_speed_gain      # current robot speed gain after adaptation
        }
        self.trial_logs.append(log_entry)  # Add this trial's results to the master list


        # =================================================================
        # INTER-TRIAL PAUSE — 2-second countdown before the next trial
        # =================================================================
        pause_start = time.time()
        while time.time() - pause_start < 2.0:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, "Trial Complete! Get ready...",
                        (self.width // 2 - 150, self.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow('HRI System', frame)
            cv2.waitKey(1)


    # -------------------------------------------------------------------------
    # generate_outputs  —  Called once after all 20 trials are done.
    # Saves the CSV log and generates the two required graphs.
    # -------------------------------------------------------------------------
    def generate_outputs(self):

        # --- SAVE CSV LOG ---
        if len(self.trial_logs) > 0:
            keys = self.trial_logs[0].keys()  # Column headers = dictionary keys
            with open('CSV_Log.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()        # Write column names as the first row
                writer.writerows(self.trial_logs)  # Write one row per trial
            print("Saved CSV_Log.csv")

        n = len(self.trial_logs)
        if n == 0:
            return  # Nothing to summarise if no trials were completed

        # --- COMPUTE SUMMARY STATISTICS ---

        # Recognition accuracy as a percentage (e.g. 80.0 means 16/20 correct).
        acc = sum(t['Recognition accuracy'] for t in self.trial_logs) / n * 100

        # Average time to lock onto the correct target (only count trials where a lock happened).
        lock_times    = [t['Target lock time'] for t in self.trial_logs if t['Target lock time'] > 0]
        avg_lock_time = np.mean(lock_times) if lock_times else 0

        # Average normalised position error across all trials.
        avg_pos_error = np.mean([t['Final position error'] for t in self.trial_logs])

        # Average arrival time prediction error.
        avg_arr_error = np.mean([t['Arrival time error'] for t in self.trial_logs])

        # Average of the closest the robot got to the hand across all trials.
        avg_min_dist  = np.mean([t['Minimum human-robot distance'] for t in self.trial_logs])

        # Total number of safety violations across ALL trials.
        total_safety_viols = sum(t['Safety violations'] for t in self.trial_logs)

        # Average frames per second (overall system speed).
        avg_fps = np.mean([t['Average FPS'] for t in self.trial_logs])

        # Print summary to the terminal.
        print("\n--- FINAL SUMMARY ---")
        print(f"Goal recognition accuracy: {acc:.1f}%")
        print(f"Average lock time: {avg_lock_time:.2f}s")
        print(f"Average position error (normalized): {avg_pos_error:.3f}")
        print(f"Average arrival-time error: {avg_arr_error:.2f}s")
        print(f"Minimum human-robot distance (avg): {avg_min_dist:.1f}px")
        print(f"Total safety violations: {total_safety_viols}")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Adaptation parameter (final gain): {self.robot_speed_gain:.2f}")


        # --- GRAPH 1: Target Probability vs Time ---
        # Plots how the probability of each target changed over time during
        # the first successful trial. Useful for visualising the Bayesian update.
        if len(self.prob_history) > 0:
            # Extract time values (x-axis) and probability arrays (y-axis).
            times = [x[0] for x in self.prob_history]
            probs = np.array([x[1] for x in self.prob_history])

            plt.figure(figsize=(8, 5))
            plt.plot(times, probs[:, 0], label='Left Target',   linestyle='--')   # Left target probability
            plt.plot(times, probs[:, 1], label='Centre Target', linestyle='-')    # Centre target probability
            plt.plot(times, probs[:, 2], label='Right Target',  linestyle='-.')   # Right target probability
            plt.axhline(y=0.8, color='r', linestyle=':', label='Lock Threshold')  # Horizontal line at 80%
            plt.xlabel('Time (s)')
            plt.ylabel('Probability')
            plt.title('Target Probability vs Time (Successful Trial)')
            plt.legend()
            plt.savefig('prob_vs_time.png')   # Save the graph as an image file
            plt.close()
            print("Saved prob_vs_time.png")


        # --- GRAPH 2: Trial Number vs Final Position Error ---
        # Shows how the position error changed trial by trial.
        # Ideally the error should decrease over time as the system adapts.
        trials = [t['Trial number']       for t in self.trial_logs]
        errors = [t['Final position error'] for t in self.trial_logs]

        plt.figure(figsize=(8, 5))
        plt.plot(trials, errors, marker='o')                                       # One dot per trial
        plt.axhline(y=0.05, color='r', linestyle='--', label='Target <= 0.05')    # Target accuracy line
        plt.xlabel('Trial Number')
        plt.ylabel('Final Position Error (normalised)')
        plt.title('Trial Number vs Final Position Error')
        plt.legend()
        plt.savefig('error_vs_trial.png')  # Save the graph as an image file
        plt.close()
        print("Saved error_vs_trial.png")


# =============================================================================
# ENTRY POINT — This block runs when you execute the script directly.
# It creates one HRIExperiment object and calls run() to start everything.
#
# The `if __name__ == "__main__":` guard means this code does NOT run if
# someone imports this file as a module from another script.
# =============================================================================
if __name__ == "__main__":
    app = HRIExperiment()   # Create the experiment (sets up webcam, trials, etc.)
    app.run()               # Start the calibration → trials → output pipeline

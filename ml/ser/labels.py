"""The 6 emotions shared by RAVDESS and CREMA-D (fixed order = class index)."""

EMOTIONS = ["neutral", "happy", "sad", "angry", "fear", "disgust"]
EMOTION_TO_ID = {e: i for i, e in enumerate(EMOTIONS)}

# RAVDESS filename field 3. NOTE: the old notebook mapped 01 -> Angry, 02 -> Happy ...
# which is wrong; that bug is the main reason it scored ~22 %.
# Official codes: 01 neutral, 02 calm, 03 happy, 04 sad, 05 angry, 06 fearful,
# 07 disgust, 08 surprised. Calm and surprised are not in CREMA-D, so they are dropped.
RAVDESS_CODES = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
}

# CREMA-D filename field 3, e.g. 1001_DFA_ANG_XX.wav
CREMAD_CODES = {
    "NEU": "neutral",
    "HAP": "happy",
    "SAD": "sad",
    "ANG": "angry",
    "FEA": "fear",
    "DIS": "disgust",
}

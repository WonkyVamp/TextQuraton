import os
import string
import joblib


MODULE_PATH = os.path.dirname(os.path.abspath(__file__))

MODEL_DATE = joblib.load(os.path.join(MODULE_PATH, "./date_model.pickle"))

ALPHA_CHAR_SET = set(string.ascii_letters)
DATE_MODEL_CHARS = []
DATE_MODEL_CHARS.extend(string.ascii_letters)
DATE_MODEL_CHARS.extend(string.digits)
DATE_MODEL_CHARS.extend(["-", "/", " ", "%", "#", "$"])

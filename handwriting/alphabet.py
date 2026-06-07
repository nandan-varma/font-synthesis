from collections import defaultdict
import numpy as np

alphabet = [
    '\x00', ' ', '!', '"', '#', "'", '(', ')', ',', '-', '.',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';',
    '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
    'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'Y',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
    'y', 'z',
]

MAX_STROKE_LEN = 1200
MAX_CHAR_LEN = 75

_alpha_to_num: defaultdict = defaultdict(int, {c: i for i, c in enumerate(alphabet)})


def encode(text: str) -> np.ndarray:
    """Encode text to an array of alphabet indices, null-terminated."""
    return np.array([_alpha_to_num[c] for c in text] + [0], dtype=np.int64)

import re


def preprocess(text):
    """Normalise text so rare specifics (URLs, long numbers, currency) become shared tokens."""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"\d{5,}", " longnumtoken ", text)
    text = re.sub(r"[£$€₹]", " currtoken ", text)
    return text

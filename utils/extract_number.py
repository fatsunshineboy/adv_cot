import re


def extract_number(text):
    """
    Extract the first integer or floating-point number from the text, supporting negative signs and commas in the thousandth place.
    If all the decimal parts are zero, return an integer. Otherwise, return a floating-point number.
    Return Err when no number is found.
    """
    pattern = r'-?\d+(?:,\d{3})*(?:\.\d+)?'
    match = re.search(pattern, text)
    if not match:
        return "Err"
    s = match.group().replace(',', '')
    if '.' in s:
        integer_part, frac_part = s.split('.', 1)
        if set(frac_part) == {'0'}:
            return int(integer_part)
        return float(s)
    return int(s)


def extract_letter(text, options=None):
    if not isinstance(text, str):
        return None

    if options is None:
        options = ['A', 'B', 'C', 'D', 'E']
    patterns = [
        r'\((\w)\)',
        r'(\w?)'
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1).upper()
            if answer in options:
                return answer

    return ""
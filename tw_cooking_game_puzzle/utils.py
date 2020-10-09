def is_whitespace(c, use_space=True):
    if (c == " " and use_space) or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def clean_str(str_in):
    str_out = ''
    for c in str_in:
        if not is_whitespace(c, use_space=False):
            str_out += c
        else:
            str_out += ' '
    return str_out
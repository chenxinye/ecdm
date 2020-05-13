from tqdm import tqdm

def symbolic_generate(length=52., start='A'):
    
    str_sqen = list()
    for i in range(length + 1):
        str_sqen.append(i)

    start = ord(start)
    letter_num = [i+1 for i in range(str_sqen[length])]
    letter = list(); i = 1
    
    for num in tqdm(letter_num + [0]):
        if i <= 26:
            letter.append(chr(64 + i))
            i = i + 1
        elif 26 < i and i <= 52:
            letter.append(chr(96 + i - 26))
            i = i + 1
        else:
            i = i - 52
    num_letter_dict = dict(zip(letter_num,letter))
    letter_num_dict = dict(zip(letter,letter_num))
    return num_letter_dict,letter_num_dict

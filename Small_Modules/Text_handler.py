#Dictionary, from letter to int
#Postive -> alphabets, Negative -> Symbols
Dict_letter_to_int = {
    " ":-1, "~":-2, "`":-3, "!":-4, "@":-5, "#":-7, "$":-8, "%":-9, "^":-10,
    "&":-11, "*":-12, "(": -13, ")": -14, "_":-15, "-":-16, "+":-17, "=":-18,
    "{":-19, "}":-20, "[": -21, "]": -22, "|":-23, "\\":-24, ":":-25, ";":-26,
    '"':-27, "'":-28, "<": -29, ">": -30, ",":-31, ".":-32, "?":-33, "/":-34,
    "‑":-35, "–":-36, "—":-37,
    "A":1, "a":2, "B":3, "b":4, "C":5, "c":6, "D":7, "d":8, "E":9, "e":10,
    "F":11, "f":12, "G":13, "g":14, "H":15, "h":16, "I":17, "i":18, "J":19, "j":20,
    "K":21, "k":22, "L":23, "l":24, "M":25, "m":26, "N":27, "n":28, "O":29, "0":30,
    "P":31, "p":32, "Q":33, "q":34, "R":35, "r":36, "S":37, "s":38, "T":39, "t":40,
    "U":41, "u":42, "V":43, "v":44, "W":45, "w":46, "X":47, "x":48, "Y":49, "y":50,
    "Z":51, "z":52, "Ä":53, "ä":54, "Ö":55, "ö":58, "Ü":57, "ü":58, "ß":59
}
#-1 just to match size " ", 0 is for unidentified
import nltk
nltk.download('wordnet', quiet=True)
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

#Given a string, convert it to NumData and use Lemmatizer
def String_function(string, size):
    string.strip(); string = lemmatizer.lemmatize(string)
    string_list_num = []
    for letter in string:
        number = Dict_letter_to_int.get(letter)
        if number is None: number = 0
        string_list_num.append(number)
    # Match size
    length = len(string_list_num)
    if length == size:
        return string_list_num
    size_diff = size - length
    if size_diff > 0:
        emp = []
        for i in range(size_diff):
            emp.append(-1)
        return emp+string_list_num
    else:
        return string_list_num[abs(size_diff):length]

#Given RawData (Dictionary), return NumData
def Text_to_Num(RawData: dict, size_I, size_O):
    #Extract into lists
    inputs = []; labels = []
    for i in RawData:
        for key, value in RawData[i].items():
            inputs.append(value["input"])
            labels.append(value["output"])
    #Convert each into num
    inputs_num = []
    for string in inputs:
        inputs_num.append(String_function(string, size_I))
    labels_num = []
    for string in labels:
        labels_num.append(String_function(string, size_O))
    NumData = [inputs_num, labels_num]
    return NumData
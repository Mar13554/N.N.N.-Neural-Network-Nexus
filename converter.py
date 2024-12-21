import nltk

try:
    nltk.data.find('wordnet')
except Exception:
    nltk.download('wordnet', quiet=True)
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

#Natural numbers for letters and numbers(operators too), everything else is negative
dict_characters = {
    " ":0, "!":-1, "?":-2, "'":-3, '"':-4, ".":-5, ",":-6, ";":-7, "&":-8,
    "(":-9, ")":-10, "[":-11, "]":-12, "#":-13, "_":-14, "$":-15,
    "a":1, "A":2, "b":3, "B":4, "c":5, "C":6, "d":7, "D":8, "e": 9, "E":10,
    "f":11, "F":12, "g":13, "G":14, "h":15, "H":16, "i":17, "I":18, "j":19, "J":20,
    "k":21, "K":22, "l":23, "L":24, "m":25, "M":26, "n":27, "N":28,"o":29, "O":30,
    "p":31, "P":32, "q":33, "Q":34, "r":35, "R":36, "s":37, "S":38, "t":39, "T":40,
    "u":41, "U":42, "v":43, "V":44, "w":45, "W":46, "x":47, "X":48, "y":49, "Y":50,
    "z":51, "Z":52, "ß":53, "ä":54, "ö":55, "ü":56, "è":57, "é":58,
    "1":59, "2":60, "3":61, "4":62, "5":63, "6":64, "7":65, "8":66,
    "9":67, "0":68, "+":69, "-":70, "*":71, "/":72, "~":73, "^":74, "%":75
}

size_list = 350

#String/strings (in a list) to list
def string_list_convert(string_list, debug_mode = None):

    #Cut string_list size and order to one string
    length_list = len(string_list)
    amount_of_words = 0; available_letters = size_list
    for i in range(0, length_list):
        #Flip value so it goes back to front, check letter amount
        if available_letters - len(string_list[length_list-1-i]) >= 0:
            available_letters -= len(string_list[length_list-1-i])
            amount_of_words += 1
        else:
            break
    #Set as one string
    string = ""
    string_list.reverse()
    for i in range(0, amount_of_words):
        string += string_list[amount_of_words-1-i]+" "
    if debug_mode:
        print(f"String input: {string}")
        print(f"Total amount of 'words': {amount_of_words}, Full string selected: {string}")
    string.strip()

    #Lemma
    string = (lemmatizer.lemmatize(string)).lower()

    #Convert with dict
    string = list(string)
    list_string1 = []
    for i in string:
        number_id = dict_characters.get(i)
        #Unrecognised letter
        if number_id is None:
            number_id = -69
        list_string1.append(number_id)

    #Match size fill with blank space 0
    list_string2 = []
    amount_space = size_list-len(list_string1)
    for i in range(amount_space):
        list_string2.append(0)
    list_string2 = list_string2 + list_string1
    if debug_mode:
        print(list_string2)
    #Final check
    while len(list_string2) > size_list:
        list_string2.pop(0)
    return list_string2

dict_numbers = dict((v, k) for k, v in dict_characters.items())

#List to string
def list_string_convert(list, debug_mode = None):
    if debug_mode:
        print(list[0])
    string = ""
    for i in list[0]:
        i = round(i)
        letter = dict_numbers.get(i)
        if letter is None:
            letter = " "
        string += letter
    return string.strip()
import json
def read_jfile(file_name):
    with open(file_name, "r") as file:
        try:
            data = json.load(file)
            return data
        except Exception:
            raise Exception("Error opening file")

def write_jfile(file_name, data):
    with open(file_name, "w") as file:
        json.dump(data, file, indent=4)
    return 0

#Change extension .json to .txt
def convert_name(file_name):
    index = 0; length = len(file_name)
    for i in range(0, length):
        if file_name[i:length] == ".json":
            index = i; break
    return file_name[0:index]+".txt"

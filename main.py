#Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
#Everything else
import os
from converter import string_list_convert, list_string_convert
from Graph_loss import loss_graph
import json, ast, time
from colorama import Fore
#Colored text
def rt(text): return Fore.RED+text+Fore.RESET
def bt(text): return Fore.BLUE+text+Fore.RESET
def gt(text): return Fore.GREEN+text+Fore.RESET
def ct(text): return Fore.CYAN+text+Fore.RESET
def lgt(text): return Fore.LIGHTGREEN_EX+text+Fore.RESET
def yt(text): return Fore.LIGHTYELLOW_EX+text+Fore.RESET
current_dir = os.path.dirname(__file__)

num_epochs = 2000
time_limit = 60
accumulate_steps = 2 #Accumulate gradients for 2 batches
loss_history = [None]
debug_mode = False
current_text_history = []
#Find data_files
try:
    with open("Settings.json", "r") as file:
        data = json.load(file)
        data_files = data["Data_Files"]
except FileNotFoundError:
    with open("Settings.json", "x") as file:
        # Create default
        data = {"Num_Epochs": 2000, "Time_Limit": 60,
                "Data_Files": ["Success.json", "Ero1_4KB.json", "Nexus.json", "books1.json", "transcriptionYT1.json"]}
        json.dump(data, file, indent=4)

print(gt("Defining model..."))
# Define
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5, output_size):
        super(NeuralNet, self).__init__()
        self.LeakyRelu = nn.LeakyReLU(negative_slope=0.75)
        self.Relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, hidden_size4)
        self.fc5 = nn.Linear(hidden_size4, hidden_size5)
        self.fc6 = nn.Linear(hidden_size5, output_size)

    def forward(self, x):
        out = self.fc1(x); out = self.LeakyRelu(out)
        out = self.fc2(out); out = self.LeakyRelu(out)
        out = self.fc3(out); out = self.LeakyRelu(out)
        out = self.fc4(out); out = self.LeakyRelu(out)
        out = self.fc5(out); out = self.LeakyRelu(out)
        out = self.fc6(out)
        return out

# Create Instance of Model (If you want to modify the input size, make sure to change list size in converter.py)
model = NeuralNet(input_size=350, hidden_size1=360, hidden_size2=370, hidden_size3=380, hidden_size4 = 370, hidden_size5 = 360, output_size=350)
# Define Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.75, nesterov=True)

def train(inputs, labels, size):
    # Convert Input and Labels
    inputs = torch.tensor(inputs, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.float)
    # Dataset
    dataset = TensorDataset(inputs, labels)
    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    #Time
    set_time = time.time()
    #Training
    for epoch in range(num_epochs):
        sum_loss = 0
        for i, (inputs, labels) in enumerate(data_loader):
            # Forward pass
            outputs = model(inputs)
            # Calculate loss
            loss = criterion(outputs, labels)
            # Backward and optimise
            loss.backward()
            if (i+1) % accumulate_steps == 0:
                optimizer.zero_grad()
                optimizer.step()
            sum_loss += float(loss)
        loss_history.append(round(sum_loss/len(inputs), 6))
        #Break for time limit
        if abs(set_time-time.time()) >= time_limit:
            break
    print(f"Reached epoch: {epoch+1}")
    print(f"Finished at {round(abs(set_time-time.time()), 6)} seconds \n")


#User
while True:
    print("Enter choice: "+rt("Exit")+", "+bt("-1")+"; "+lgt("Learn")+", "+bt("1")+"; "+"Chat History, "+bt("2")+"; "+ct("Chat")+", "+bt("3")+"; Loss history graph, "+bt("4"))
    choice = int(input(yt("Save parameters")+", "+bt("5")+"; "+yt("Load parameters")+", "+bt("6")+"; "+yt("settings")+", "+bt("7")+"\n"))
    match choice:
        #Exit
        case -1:
            print(rt("Exit -1"))
            break
        #Learn
        case 1:
            #Find file
            file_chosen = int(input("Enter file id: "))
            try: file_chosen = data_files[file_chosen-1]
            except IndexError: raise Exception("File from id not found")

            #Find list
            list_data_chosen = input("Enter list/lists id for training (Format: 1, [1, 2], or 1.0 for all): ")
            file_path = os.path.join(current_dir, "TestData")
            file_path = os.path.join(file_path, file_chosen)
            try:
                with open(file_path, "r") as file:
                    list_arrays = json.load(file)
            except FileNotFoundError:
                raise Exception("File not found")
            loss_history.clear()

            #Learn from 1 list
            if isinstance(ast.literal_eval(list_data_chosen), int):
                try:
                    list_array = list_arrays[list_data_chosen]
                    if debug_mode:
                        print(list_array)
                    inputs = []; labels = []
                    for key, value in list_array.items():
                        inputs.append(string_list_convert([value["input"]], debug_mode))
                        labels.append(string_list_convert([value["output"]], debug_mode))
                    print(f"Training... file: {file_chosen}")
                    train(inputs, labels, len(list_array))
                except Exception:
                    print(rt("list id doesn't exist or number error (check input size)"))
            #Learn from multiple lists
            elif isinstance(ast.literal_eval(list_data_chosen), list):
                inputs = []; labels = []
                for i in ast.literal_eval(list_data_chosen):
                    try:
                        for key, value in list_arrays[str(i)].items():
                            inputs.append(string_list_convert([value["input"]]))
                            labels.append(string_list_convert([value["output"]]))
                    except Exception:
                        print(rt("A list id doesn't exist or input size error"))
                        break
                try:
                    print(f"Training... file: {file_chosen}")
                    train(inputs, labels, len(inputs))
                except Exception:
                    print(rt("Training error"))
            # Learn from every list
            elif isinstance(ast.literal_eval(list_data_chosen), float):
                inputs = []; labels = []
                print("Packaging inputs and outputs...")
                for i in list_arrays:
                    try:
                        for key, value in list_arrays[i].items():
                            inputs.append(string_list_convert([value["input"]]))
                            labels.append(string_list_convert([value["output"]]))
                    except Exception:
                        print(rt("Input size error"))
                        break
                print(f"Training... file: {file_chosen}")
                train(inputs, labels, len(inputs))
            else:
                print(f"list/lists id incorrect format: {list_data_chosen}")
        #Load or save text history
        case 2:
            chat_selected = int(input("Which chat would you like to select? 1, 2, 3 Please enter integer: "))
            chats = ["chat_history_1.txt", "chat_history_2.txt", "chat_history_3.txt"]
            file_path = os.path.join(current_dir, "Text_History")
            try:
                file_path = os.path.join(file_path, chats[chat_selected-1])
            except Exception:
                raise Exception(f"Chat {chat_selected}, not found")
            chat_access = int(input("What would you like to do with it? Load, 1; Save, 2; Delete, 3: "))
            #Test if file exists
            try:
                with open(file_path, "x") as file:
                    print("File not found, creating it...")
            except Exception:
                print("File exists")
            #Manage file
            match chat_access:
                case 1:
                    with open(file_path, "r") as file:
                        current_text_history = ast.literal_eval(file.read())
                        for i in current_text_history:
                            print(i)
                case 2:
                    with open(file_path, "w") as file:
                        file.write(str(current_text_history))
                        print("Chat Saved!")
                case 3:
                    with open(file_path, "w") as file:
                        print("Chat deleted!")

        #Test
        case 3:
            print("Enter -1, to exit chat!")
            while True:
                test_input = input("Enter text: ")
                try:
                    if ast.literal_eval(test_input) == -1:
                        print("Chat exit complete \n")
                        break
                except Exception:
                    pass
                #Save in chat history
                current_text_history.append(test_input)
                #Set input to right type (list of ints)
                inputs = [string_list_convert(current_text_history.copy(), debug_mode)]
                inputs = torch.tensor(inputs, dtype=torch.float)
                #Get output and convert to string
                output = ((model(inputs)).detach().numpy())
                if debug_mode:
                    print(f"Output numbers: {output}")
                outputs = list_string_convert(output)
                print(f"Bot: {outputs}")
                #Save in chat history
                current_text_history.append(outputs)

        #Graph
        case 4:
            if loss_history[0] is None:
                print(rt("No recent history"))
            else:
                loss_graph(loss_history)

        #Save parameters
        case 5:
            torch.save(model.state_dict(), "data.pth")
            print(gt("Saved! \n"))
        #Load parameters
        case 6:
            model.load_state_dict(torch.load("data.pth", weights_only=True))
            print(gt("Loaded! \n"))

        case 7:
            settings_choice = input(bt("1")+", Load settings; "+bt("2")+", Epochs; "+bt("3")+", Time_limit; "+bt("4")+", data_files: ")
            try:
                settings_choice = int(settings_choice)
            except TypeError:
                pass
            #Check file exists
            try:
                with open("Settings.json", "x") as file:
                    #Create default
                    data = {"Num_Epochs":2000, "Time_Limit":60, "Data_Files":["Success.json", "Ero1_4KB.json", "Nexus.json", "books1.json", "transcriptionYT1.json"]}
                    json.dump(data, file, indent=4)
            except FileExistsError:
                pass
            match settings_choice:
                case 1:
                    #Load settings
                    with open("Settings.json", "r") as file:
                        data = json.load(file)
                        num_epochs = data["Num_Epochs"]
                        time_limit = data["Time_Limit"]
                        data_files = data["Data_Files"]
                        print(f"Settings loaded! Epochs: {num_epochs}, Time limit: {time_limit}s\n")
                case 2:
                    #Epochs
                    try:
                        num_epochs = int(input(f"Current Epoch: {num_epochs}, Set Epoch (amount of times trained): "))
                        print(f"Epochs set to {num_epochs} \n")
                        data = {"Num_Epochs":num_epochs, "Time_Limit":time_limit, "Data_Files":data_files}
                        with open("Settings.json", "w") as file:
                            json.dump(data, file, indent=4)
                    except Exception:
                        print("Error: non-integer in input \n")
                case 3:
                    # Time Limit
                    try:
                        time_limit = int(input(f"Current time limit: {time_limit}, Set time limit: "))
                        print(f"Time Limit set to {time_limit} seconds \n")
                        data = {"Num_Epochs": num_epochs, "Time_Limit": time_limit, "Data_Files":data_files}
                        with open("Settings.json", "w") as file:
                            json.dump(data, file, indent=4)
                    except Exception:
                        print("Error: non-integer in input \n")
                case _:
                    print("Error: Enter an integer to select settings \n")

        #Debug mode
        case -99:
            if debug_mode:
                debug_mode = False
                print(ct("Debug mode off"))
            else:
                debug_mode = True
                print(ct("Debug mode on"))
        case _:
            print(rt(f"""Command "{choice}" invalid, please try again. \n"""))
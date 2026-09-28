#Imports
import os, time, ast, sys
from Small_Modules.json_func import read_jfile, convert_name
from Small_Modules.Folder_handle import check_in_folder
from Small_Modules.Text_handler import Text_to_Num
from C_pipe import Compile_data_for_cpp, Run_cpp_file
#Function from pyinstaller usage
def source_path(path):
	try:
		base_path = os.path.dirname(sys._MEIPASS)
	except Exception:
		base_path = path
	return base_path

#Paths and files
directory = source_path(os.path.dirname(__file__))
all_files = check_in_folder(directory)

def check_needed_files():
	if "TestData" not in all_files: raise Exception(f"TestData folder not found. Path:{directory}")
	if "Cached_data_files" not in all_files: raise Exception(f"Cached_data_files folder not found. Path:{directory}")
	return os.path.join(directory, "TestData"), os.path.join(directory, "Cached_data_files")

#Dict Files
def Data_Files(TestData_folder_path):
	Dict_files = {}
	Data_files = check_in_folder(TestData_folder_path)
	for i in range(0, len(Data_files)):
		Data_file_path = os.path.join(TestData_folder_path, Data_files[i])
		Dict_files.update({i: [Data_files[i],
						 f"File size: {round(os.path.getsize(Data_file_path)/1000, 4)} KB",
						 f"Date modified: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(Data_file_path)))}"]})
	return Dict_files

#Model variables and data
model_info = None
saved_pm = False
input_data = None
output_data = None

def var_reset():
	global model_info, saved_pm, input_data, output_data
	model_info = None
	saved_pm = False
	input_data = None
	output_data = None

#Check for save
def check_save():
	global saved_pm, model_info
	if "Main_model.txt" in all_files:
		choice = input("Would you like to load Model data? Y/N \n")
		if choice == "Y":
			with open(os.path.join(directory, "Main_model.txt"), "r") as model_file:
				model_info = ast.literal_eval(model_file.read())
			saved_pm = True

		elif choice == "N":
			pass
		else:
			print("Please enter either 'Y' for yes, or 'N' for no")

#Select file
def select_file(TestData_folder_path):
	Dict_files = Data_Files(TestData_folder_path)
	print("Files:")
	for k, v in Dict_files.items():
		print(f"{k}: {v}")
	choice = int(input(f"Enter data file you want to use: \n"))
	file_name_chosen = (Dict_files.get(choice))[0]
	if file_name_chosen is None:
		print(f"Please enter a number included, not '{choice}'")
		return -1
	return file_name_chosen

#Check if file is also cached:
def select_cached_file(file_name_chosen: str, Cached_data_files_path):
	global input_data, output_data
	Cached_files = check_in_folder(Cached_data_files_path)
	if convert_name(file_name_chosen) in Cached_files:
		file_name_cached_path = os.path.join(Cached_data_files_path, convert_name(file_name_chosen))
		print(f"Cached file found: Last modified: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(file_name_cached_path)))}")
		choice = input("Would you like to use cached file? Y/N \n")
		match choice:
			case "Y":
				with open(file_name_cached_path, "r") as cached_file:
					cached_data = ast.literal_eval(cached_file.read())
					input_data = cached_data[0]
					output_data = cached_data[1]
					return 0
			case "N":
				return 0
			case _:
				print("Please enter 'Y' (yes) or 'N' (no).")
				return -1

#Run
def run():
	global model_info, input_data, output_data, saved_pm
	TestData_folder_path, Cached_data_files_path = check_needed_files()
	check_save()
	file_name = select_file(TestData_folder_path)
	if file_name == -1: return -1
	select_cached_file(file_name, Cached_data_files_path)
	if model_info is None: #New Model
		layer_info = ast.literal_eval(input("Enter the amount of nodes per layer in the form of a python list, e.g. '[2, 2, 2]': \n"))
		model_info = [layer_info]
	if input_data is None: #Get data and cache it
		#Get data
		print("Converting data...")
		Raw_data = read_jfile(os.path.join(TestData_folder_path, file_name))
		Num_data = Text_to_Num(Raw_data, model_info[0][0], model_info[0][-1])
		input_data = Num_data[0]
		output_data = Num_data[1]
		#Cache data
		with open(os.path.join(Cached_data_files_path, convert_name(file_name)), "w") as cached_file:
			cached_file.write(str(Num_data))
	epoch = int(input("Enter epochs (amount of times the program will train for): "))
	print("Packaging input...")
	packaged_input = Compile_data_for_cpp(model_info, saved_pm, input_data, output_data, len(input_data), epoch)
	print("Running...")
	output = ast.literal_eval(Run_cpp_file(packaged_input)) #Returns [Avg Loss data, Model info, Model parameters]
	#Save
	with open(os.path.join(directory, "Main_model.txt"), "w") as model_file:
		model_file.write(str([output[1], output[2]]))
	print(f"Average MAE Loss rate: {output[0]}")

def custom_use():
	manual_command = input("Enter full command: (For more information check the documentation for the format):\n")
	manual_command = ast.literal_eval(f"'{manual_command}'")
	output = str(Run_cpp_file(manual_command))
	return output

#User
if "Y" == input("Would you like to enter the full command to the .exe directly? (Not recommended) Y/N: "):
	while (True):
		choice = int(input("Enter '-1' to exit, enter any other integer to continue: "))
		if choice == -1:
			break
		else:
			print(custom_use())
else:
	run()
	var_reset()
	while (True):
		if "Y" == input("Would you like to continue? Y/N: "):
			run()
			var_reset()
		else:
			break
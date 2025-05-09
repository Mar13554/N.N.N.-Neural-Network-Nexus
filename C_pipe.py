import subprocess, struct

def write_binary_data(filename, data):
    with open(filename, 'wb') as f:  # 'wb' for write binary
        for value in data:
            if type(value) == int:
                f.write(struct.pack('i', value))
            elif type(value) == float:
                f.write(struct.pack('d', value))  # 'd' for double
            else:
                print("Error unsupported type?")

def prepare_parameters(Model_info: list):
    #Get formatted
    Data_to_write = []
    for i_layer in range(1, len(Model_info[1])):
        for i_node in range(0, len(Model_info[1][i_layer])):
            parameters = Model_info[1][i_layer][i_node]
            Data_to_write += [i_layer - 1, i_node, len(parameters)-1]
            for pm in parameters:
                Data_to_write.append(pm)
    #Add to .bin
    write_binary_data("Transfer.bin", Data_to_write)


def Compile_data_for_cpp(Model_info: list, saved_parameters: bool, inputs: list, outputs: list, io_amount: int, epoch: int):
    # First part
    input_command = ""
    for i in Model_info[0]:
        input_command += str(i) + " "
    input_command += "\n"
    # Commands
    # Parameters
    if saved_parameters:
        input_command += "p\n"
        prepare_parameters(Model_info)

    # io_amount
    input_command += "d " + str(io_amount) + "\n"
    # Input
    input_command += "i\n"
    for i in inputs:
        for j in i:
            input_command += str(j) + " "
        input_command += "\n"
        # Label
    input_command += "o\n"
    for i in outputs:
        for j in i:
            input_command += str(j) + " "
        input_command += "\n"
        # Train
    input_command += "t " + str(epoch) + "\n"
    # Break
    input_command += "b\n"

    return input_command

def Run_cpp_file(command: str):
   cpp_file = "C++_N.N.N.exe"

   process = subprocess.Popen([cpp_file], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

   # Encode input data as bytes
   input_bytes = command.encode("utf-8")

   # Send input data to the program's standard input
   stdout, stderr = process.communicate(input=input_bytes)
   # Check for errors
   if stderr:
       print(f"Error running C++ program: {stderr.decode('utf-8')}")
       return None

   # Decode output from bytes to string
   output = stdout.decode("utf-8")
   return output

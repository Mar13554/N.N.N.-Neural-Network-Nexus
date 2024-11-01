#Neural Network, V.0.1 (pre-alpha), official release.

from Imports.Functions import check_range
from Imports.Functions import calculate
from Imports.Functions import sigmoid
import random
import time

#Info:
#Test data [[0, 0, 0, 0], [1, 1, 1, 1], [0.5, 0.5, 0.5, 0.5], [0.6, 0.5, 0.4, 0.5], [0.2, 0.2, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1], [0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4], [0.7, 0.7, 0.7, 0.7]]
#limit, 3 (tolerance 0.1); limit 4, (tolerance 0.15); limit, 5-6 (tolerance 0.2); limit ?>=10 (tolerance 0.3)
#As in "doable" but may take a while.

#Values by (change):
C = 0.05
tolerance = 0.1
time_limit = 60 #1 minutes default

#Defining node class.
class FunctionNode_1:
    bias = random.randint(-1, 1)
    a1 = random.randint(-2, 2)
    a2 = random.randint(-2, 2)
    a3 = random.randint(-2, 2)
    result = None
class FunctionNode_2:
    bias = random.randint(-1, 1)
    a1 = random.randint(-2, 2)
    a2 = random.randint(-2, 2)
    result = None
class OutputNode:
    bias = random.randint(-1, 1)
    a1 = random.randint(-2, 2)
    a2 = random.randint(-2, 2)
    result = None

#Define nodes
    #1-Function nodes [2]
F1_Node1 = FunctionNode_1()
F1_Node2 = FunctionNode_1()
    #2-Function nodes [2]
F2_Node1 = FunctionNode_2()
F2_Node2 = FunctionNode_2()
    #Output node [1]
O_Node1 = OutputNode()
I = []

def func_calc(i1, i2, i3):
    F1_Node1.result = sigmoid(calculate(F1_Node1, i1, i2, i3))
    F1_Node2.result = sigmoid(calculate(F1_Node2, i1, i2, i3))
    F2_Node1.result = sigmoid(calculate(F2_Node1, F1_Node1.result, F1_Node2.result))
    F2_Node2.result = sigmoid(calculate(F2_Node2, F1_Node1.result, F1_Node2.result))
    O_Node1.result = sigmoid(calculate(O_Node1, F1_Node1.result)) 

option = ""
#Run
while True:
    print("Options: \n1, train from one data set; 2, train from multiple data sets; 3, test data set;")
    print("4, Save parameters; 5, Import; 6, Clear data; 7, Set time limit for training (approximate).")
    option = int(input("Choose option: "))  # 1 One data, 2 Table, 3 test.
    if option == 1:
        I.clear()
        start_time = time.process_time()
        # Input nodes [3]
        for i in range(0, 3):
            I.append(check_range(input(f"Node {i + 1},Value: ")))
        # Expected value
        Expected_value = check_range(input("Input expected value: "))
        while Expected_value != O_Node1.result:
            #Forward pass
                #Calculate values
            func_calc(I[0], I[1], I[2])
            #Error rate
            Error = abs(Expected_value - O_Node1.result)
            # Determine results
            print("Result:", O_Node1.result)
            print("Difference:", Error, "\n")
            #Tolerence
            if Error < tolerance:
                print(f"Adjusting done, training done: {round(time.process_time()-start_time, 3)} seconds. \n")
                break
            #Adjust parameters
            for i in range(1, 4):
                a = random.randint(1, 17)
                if O_Node1.result > Expected_value:
                    c = -C
                elif O_Node1.result < Expected_value:
                    c = C
                else:
                    c = 0
                #Amplify change
                c *= (100*Error)/i
                match a:
                    case 1:
                        F1_Node1.bias += c
                    case 2:
                        F1_Node1.a1 += c
                    case 3:
                        F1_Node1.a2 += c
                    case 4:
                        F1_Node1.a3 += c
                    case 5:
                        F1_Node2.bias += c
                    case 6:
                        F1_Node2.a1 += c
                    case 7:
                        F1_Node2.a2 += c
                    case 8:
                        F1_Node2.a3 += c
                    case 9:
                        F2_Node1.bias += c
                    case 10:
                        F2_Node1.a1 += c
                    case 11:
                        F2_Node1.a2 += c
                    case 12:
                        F2_Node2.bias += c
                    case 13:
                        F2_Node2.a1 += c
                    case 14:
                        F2_Node2.a2 += c
                    case 15:
                        O_Node1.bias += c
                    case 16:
                        O_Node1.a1 += c
                    case 17:
                        O_Node1.a2 += c
    elif option == 2:
        table = eval(input("Add list with format: [[1, 2, 3, 4],...] 1-3 input, 4 expected output: \n"))
        length = len(table)
        correct_values = False
        start_time = time.process_time()
        while (correct_values is False) and ((time.process_time()-start_time)<time_limit):
            #Loop adjust
            for i in range(0, length):
                while table[i][3] != O_Node1.result:
                    # Forward pass
                        # Calculate values
                    func_calc(table[i][0], table[i][1], table[i][2])
                    # Error rate
                    Error = abs(table[i][3] - O_Node1.result)
                    #Determine results
                    print("Result:", round(O_Node1.result, 12))
                    print("Difference:", round(Error, 12), "\n")
                    #Tolerance
                    if Error < tolerance:
                        print("Adjusting done \n")
                        break
                    # Adjust parameters (Ideal amount already found)
                    for j in range(1, 4):
                        a = random.randint(1, 17)
                        if O_Node1.result > table[i][3]:
                            c = -C
                        elif O_Node1.result < table[i][3]:
                            c = C
                        else:
                            c = 0
                        # Amplify change
                        c *= (100*Error)/j
                        match a:
                            case 1:
                                F1_Node1.bias += c
                            case 2:
                                F1_Node1.a1 += c
                            case 3:
                                F1_Node1.a2 += c
                            case 4:
                                F1_Node1.a3 += c
                            case 5:
                                F1_Node2.bias += c
                            case 6:
                                F1_Node2.a1 += c
                            case 7:
                                F1_Node2.a2 += c
                            case 8:
                                F1_Node2.a3 += c
                            case 9:
                                F2_Node1.bias += c
                            case 10:
                                F2_Node1.a1 += c
                            case 11:
                                F2_Node1.a2 += c
                            case 12:
                                F2_Node2.bias += c
                            case 13:
                                F2_Node2.a1 += c
                            case 14:
                                F2_Node2.a2 += c
                            case 15:
                                O_Node1.bias += c
                            case 16:
                                O_Node1.a1 += c
                            case 17:
                                O_Node1.a2 += c
            #Check
            print("Checking...")
            local_check = True
            for i in range(0, length):
                # Forward pass
                    # Calculate values
                func_calc(table[i][0], table[i][1], table[i][2])
                # Error rate
                Error = abs(table[i][3] - O_Node1.result)
                if Error > tolerance:
                    local_check = False
                    print("Failed try again")
                    break
            if local_check is True:
                print(f"Success! Training time: {round(time.process_time()-start_time, 3)} seconds. \n")
                break

    elif option == 3:
        # Input nodes [3]
        I.clear()
        for i in range(0, 3):
            I.append(check_range(input(f"Node {i + 1},Value: ")))
        # Forward pass
            # Calculate values
        func_calc(I[0], I[1], I[2])
        #Determine results
        print("Result:", round(O_Node1.result, 12))
        print("Result (value) of nodes:")
        print(f"F1-1: {round(F1_Node1.result, 18)}, F1-2: {round(F1_Node2.result, 18)}")
        print(f"F2-1: {round(F2_Node1.result, 18)}, F2-2: {round(F2_Node2.result, 18)}")
        print("Biases of nodes:")
        print(f"F1-1: {round(F1_Node1.bias, 18)}, F1-2: {round(F1_Node2.bias, 18)}")
        print(f"F2-1: {round(F2_Node1.bias, 18)}, F2-2: {round(F2_Node2.bias, 18)}\n")
    elif option == 4:
        #Store data
        File = open("store.txt", "w")
        File.write(f"[[{F1_Node1.a1}, {F1_Node1.a2}, {F1_Node1.a3}, {F1_Node1.bias}],")
        File.close()
        File = open("store.txt", "a")
        File.write(f"[{F1_Node2.a1}, {F1_Node2.a2}, {F1_Node2.a3}, {F1_Node2.bias}],")
        File.write(f"[{F2_Node1.a1}, {F2_Node1.a2}, {F2_Node1.bias}],")
        File.write(f"[{F2_Node2.a1}, {F2_Node2.a2}, {F2_Node2.bias}],")
        File.write(f"[{O_Node1.a1}, {O_Node1.a2}, {O_Node1.bias}]]")
        File.close()
        print("Data saved!")
    elif option == 5:
        #Import data
        File = open("store.txt", "r")
        list_read = eval(File.read())
        print(list_read)
        if list_read != "":
            F1_Node1.a1 = list_read[0][0]; F1_Node1.a2 = list_read[0][1]; F1_Node1.a3 = list_read[0][2]; F1_Node1.bias = list_read[0][3]
            F1_Node2.a1 = list_read[1][0]; F1_Node2.a2 = list_read[1][1]; F1_Node2.a3 = list_read[1][2]; F1_Node2.bias = list_read[1][3]
            F2_Node1.a1 = list_read[2][0]; F2_Node1.a1 = list_read[2][1]; F2_Node1.bias = list_read[2][2]
            F2_Node2.a1 = list_read[3][0]; F2_Node2.a1 = list_read[3][1]; F2_Node2.bias = list_read[3][2]
            O_Node1.a1 = list_read[4][0]; O_Node1.a2 = list_read[4][1]; O_Node1.bias = list_read[4][2]
            print("Import success!")
        else:
            print("Error, empty file.")
        File.close()
    elif option == 6:
        #Clear data
        File = open("store.txt", "w")
        File.write("")
        File.close()
        print("File cleared!")
    elif option == 7:
        time_limit = int(input("Enter time limit of learning: "))
        print(f"{time_limit} seconds chosen.")
    else:
        print("Option not recognised, exiting")
        break
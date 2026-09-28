using namespace std;
#include <iostream>
#include <random>
#include <chrono>
#include <vector>
#include <sstream>
#include <string>
#include <cmath>
#include <fstream>
#include <queue>

//Math
double leaky_relu(double x){
    if (x > 0)
        return x;
    else
        return x * 0.25;
}

double MAE_loss_function(vector<double> output, vector<double> expected_output){
    double loss = 0;
    for (int i = 0; i < output.size(); i++){
        loss += abs(output.at(i) - expected_output.at(i));
    }
    return loss;
}

//Classes
class Node {
private:
    //Define
    double* weights;
    int size_weights;
    double bias;
    double value;

public:
    //Inititalze
    Node(double x, double* y, int z) {
        bias = x;
        weights = y;
        size_weights = z;
    }

    //Modify
    void modify(double x, double* y) {
        bias = x;
        Clean();
        weights = y;
    }
    void Modify(double pm, int idx) {
        if (idx == -1) {
            bias = pm;
        }
        else {
            weights[idx] = pm;
        }
    }
    void add(double pm, int idx) {
        if (idx == -1) {
            bias += pm;
        }
        else {
            weights[idx] += pm;
        }
    }

    //Update
    double update(vector<double> values) {
        if (values.size() != size_weights) {
            std::cout << "-1";
            throw runtime_error("Size not equal");
        }
        else {
            value = 0;
            for (int i = 0; i < size_weights; i++) {
                value += weights[i] * values.at(i);
            }
            value += bias;
            return leaky_relu(value);
        }
    }
    //Get value
    double get_value() {
        return value;
    }

    //Info
    int get_size() {
        return size_weights;
    }
    double* get_weights() {
        return weights;
    }
    double get_weights(int idx) {
        return weights[idx];
    }
    double get_bias() {
        return bias;
    }

    //Clean
    void Clean() {
        if (weights != nullptr) {
            delete[] weights;
            weights = nullptr;
        }
    }
};

//Random numbers
int Max_Number = 10;
random_device rd;
mt19937 gen(rd());
uniform_int_distribution<> State(0, 1);
uniform_int_distribution<> Range(0, Max_Number);
double random_number() {
    int sign = State(gen);
    int x1 = Range(gen);
    int x2 = Range(gen);
    if (x2 == 0)
        x2 = 1;
    double x = x1 / x2;
    if (sign == 1)
        return -x;
    return x;
}

double* generate_random_weights(int weight_amount) {
    double* weights = new double[weight_amount];
    for (int i = 0; i < weight_amount; i++) {
        weights[i] = random_number();
    }
    return weights;
}

vector<Node> generate_layer(int node_amount, int weight_amount) {
    vector<Node> layer;
    double random_bias = rand() % 21;
    for (int i = 0; i < node_amount; i++) {
        layer.push_back(Node(random_number(), generate_random_weights(weight_amount), weight_amount));
    }
    return layer;
}

class Model {
    private:
        vector<vector<Node>> Layers;
        int input_needed;
    public:
        //Format of {int layer_0, int layer_1, int layer_2, ...}
        //layer_0 (inputs) is never created!!
        Model(vector<int> layer_amounts) {
            for (int i = 1; i < layer_amounts.size(); i++){
                Layers.push_back(generate_layer(layer_amounts.at(i), layer_amounts.at(i - 1)));
            }
            input_needed = layer_amounts[0];
        }

        vector<double> Forward_pass(vector<double> values) {
            //Check size
            if (values.size() != input_needed) {
                std::cout << "-1";
                throw runtime_error("Size not equal");
            }
            vector<double> next_values;
            //Calculate each layer
            for (int idx = 0; idx < Layers.size(); idx++) {
                //Update layer
                for (int node_idx = 0; node_idx < Layers[idx].size(); node_idx++) {
                    Layers[idx][node_idx].update(values);
                    next_values.push_back(Layers[idx][node_idx].get_value());
                }
                values = next_values;
                next_values.clear();
            }
            return values;
        }
        void load_parameters(int layer, int node, double* weights, double bias) {
            Layers[layer][node].modify(bias, weights);
        }
        //Return layers address
        vector<vector<Node>>* return_layer_pointers() {
            return &Layers;
        }

        //Return length of inputs needed
        int length_inputs_needed() {
            return input_needed;
        }

        //Return length of outputs expected
        int length_outputs_expected() {
            return Layers[Layers.size()-1].size();
        }

        //Clear pointers in nodes
        void finish() {
            //Each layer
            for (int idx = 0; idx < Layers.size(); idx++) {
                for (int node_idx = 0; node_idx < Layers[idx].size(); node_idx++) {
                    Layers[idx][node_idx].Clean();
                }
            }
        }
};

struct container_gradient_node {
    vector<double> weight_gradients;
    double bias_gradient;
};

//Get amount of layers
vector<int> get_amount_layers() {
    vector<int> amount_layers_vector;
    string line;
    getline(cin, line); // Read a line of input
    stringstream ss(line);
    int amount_in_layer;

    while (ss >> amount_in_layer) {
        amount_layers_vector.push_back(amount_in_layer);
    }

    return amount_layers_vector;
}
//Formatted list of inputs as python [weights-0,..., weights-n, bias]
//Given ptr Main_Model parameters of model accordingly by further inputs in the order of layer -> node
void load_data_p(Model* Main_Model) {
    int layer; int node; int amount; //Layer here is -1 of layer in python form
    string line;
    getline(cin, line);
    stringstream ss(line);
    while (true) {
        ss >> layer; ss >> node; ss >> amount;
        //Weights
        double* weights = new double[amount];
        for (int i = 0; i < amount; i++) {
            ss >> weights[i];
        }
        //Bias
        double bias;
        ss >> bias;
        if (ss.fail()) {
            ss.clear();
            cout << "Error: Failed type when loading parameters.";
            throw runtime_error("No.");
        }
        Main_Model->load_parameters(layer, node, weights, bias);
        if (ss.eof()) {
            break;
        }
    }
}
struct DataChunk {
    int int1;
    int int2;
    int int3;
    vector<double> floats;
    int state = 3; //3 empty, 2, 1, 0 full requires floats
    int float_needed = 0; //Counter, set later
};
//New void load_data with .bin
void Load_Data_P(Model* Main_Model) {
    ifstream inputFile("Transfer.bin", ios::binary);
    vector<int> ints;
    vector<double> doubles;
    if (!inputFile.is_open()) {
        cerr << "Error: Could not open binary file." << endl;
    }
    // Read the data according to the expected format
    queue<DataChunk> Chunks;
    DataChunk Temp_Chunk;
    int int_value;
    double double_value;

    while (true) {
        if (Temp_Chunk.state > 0) {
            if (inputFile.read(reinterpret_cast<char*>(&int_value), sizeof(int_value))) {
                switch (Temp_Chunk.state) {
                case 1:
                    Temp_Chunk.int3 = int_value;
                    Temp_Chunk.float_needed = int_value + 1;
                    Temp_Chunk.state--;
                    break;
                case 2:
                    Temp_Chunk.int2 = int_value;
                    Temp_Chunk.state--;
                    break;
                case 3:
                    Temp_Chunk.int1 = int_value;
                    Temp_Chunk.state--;
                    break;
                }
            }
            else {
                if (inputFile.eof()) {
                    break; // End of file
                }
                else {
                    std::cerr << "Error with .bin, wrong format (int)" << std::endl;
                    throw std::runtime_error("Error with .bin, wrong format");
                }
            }
        }
        else {
            if (inputFile.read(reinterpret_cast<char*>(&double_value), sizeof(double_value))) {
                if (Temp_Chunk.float_needed > 0) {
                    Temp_Chunk.floats.push_back(double_value);
                    Temp_Chunk.float_needed--;
                    if (Temp_Chunk.float_needed == 0) {
                        Chunks.push(Temp_Chunk);
                        Temp_Chunk = DataChunk();
                    }
                }
                else {
                    std::cerr << "Error with .bin, wrong format (double)" << std::endl;
                    throw std::runtime_error("Error with .bin, wrong format");
                }
            }
            else {
                if (inputFile.eof()) {
                    break;
                }
                else {
                    std::cerr << "Error with .bin, wrong format (double)" << std::endl;
                    throw std::runtime_error("Error with .bin, wrong format");
                }
            }
        }
    }

    inputFile.close();
    //Change
    while (!Chunks.empty()) {
        DataChunk Temp_Chunk = Chunks.front();
        int layer; int node; int amount; //Layer here is -1 of layer in python form
        layer = Temp_Chunk.int1;
        node = Temp_Chunk.int2;
        amount = Temp_Chunk.int3;
        double* weights = new double[amount];
        for (int i = 0; i < amount; i++) {
            weights[i] = Temp_Chunk.floats.at(i);
        }
        //Bias
        double bias = Temp_Chunk.floats.at(amount);
        Chunks.pop();
        Main_Model->load_parameters(layer, node, weights, bias);
    }
}

//Returns a pointer to a vector<double> (single input or output)
vector<double>* get_data_io(const string& line) {
    vector<double>* ptr_vector_io = new vector<double>();
    stringstream ss(line);
    double temp_value;
    while (ss >> temp_value) {
        ptr_vector_io->push_back(temp_value);
    }
    return ptr_vector_io;
}

//Returns a vector pointer to pointers to a vector (list of inputs or outputs)
void fill_io_pointers(vector<vector<double>*>* io, int io_amount, int length) {
    for (int i = 0; i < io_amount; i++) {
        string line;
        getline(cin, line);
        io->push_back(get_data_io(line));
    }
}
//Constants
double h = pow(10, -8);
double learning_rate = pow(10, -4);
//Data
double average_loss(vector<double> temp_losses) {
    double sum = 0; int vector_size = temp_losses.size();
    for (int i = 0; i < vector_size; i++){
        sum += temp_losses.at(i);
    }
    return (sum / vector_size);
}
vector<double> temp_losses;
vector<double> avg_losses;
void train(Model* ptr_Main_model, vector<vector<double>*>* dataset_input, vector<vector<double>*>* dataset_output, int epochs, int io_amount){
    using std::cout;
    vector<vector<Node>>* ptr_layer = (ptr_Main_model->return_layer_pointers());
    int const layer_amount = ptr_layer->size();
    for (int i = 1; i <= epochs; i++) {
        for (int idx = 0; idx < io_amount; idx++) { //Using (h=1x10*{-8}) (L(x+h, ...)-L(x, ...))/h for gradient of x
            //Calculate for L(x)
            double loss = MAE_loss_function(ptr_Main_model->Forward_pass(*((*dataset_input)[idx])), *((*dataset_output)[idx]));
            temp_losses.push_back(loss);
            //Calculate each gradient (vector of L(x+h) for each layer)
            vector<vector<container_gradient_node>> gradients;
            for (int i_layer = 0; i_layer < layer_amount; i_layer++) {
                vector<container_gradient_node> layer_gradients;
                int node_amount = (*ptr_layer)[i_layer].size();
                for (int i_node = 0; i_node < node_amount; i_node++) {
                    int weights_amount = (*ptr_layer)[i_layer][i_node].get_size();
                    vector<double> weight_gradients;
                    for (int i_weight = 0; i_weight < weights_amount; i_weight++) {
                        //Save
                        double saved_parameter = (*ptr_layer)[i_layer][i_node].get_weights(i_weight);
                        //Modify by h
                        (*ptr_layer)[i_layer][i_node].Modify(saved_parameter + h, i_weight);
                        //Run and save value
                        double n_loss = MAE_loss_function(ptr_Main_model->Forward_pass(*((*dataset_input)[idx])), *((*dataset_output)[idx]));
                        //Compute gradient and modify with learning rate 
                        double gradient = -learning_rate * ((n_loss - loss) / h);
                        weight_gradients.push_back(gradient);
                        //Change weight back
                        (*ptr_layer)[i_layer][i_node].Modify(saved_parameter, i_weight);
                    }
                    //Compute for bias gradient
                    //Save
                    double saved_parameter = (*ptr_layer)[i_layer][i_node].get_bias();
                    //Modify
                    (*ptr_layer)[i_layer][i_node].Modify(saved_parameter + h, -1);
                    //Run and save value
                    double n_loss = MAE_loss_function(ptr_Main_model->Forward_pass(*((*dataset_input)[idx])), *((*dataset_output)[idx]));
                    //Compute gradient and modify with learning rate 
                    double gradient = -learning_rate * ((n_loss - loss) / h);
                    //Change bias back
                    (*ptr_layer)[i_layer][i_node].Modify(saved_parameter, -1);
                    //Update struc and add to vector
                    container_gradient_node temp_gradient_node;
                    temp_gradient_node.weight_gradients = weight_gradients;
                    temp_gradient_node.bias_gradient = gradient;
                    layer_gradients.push_back(temp_gradient_node);
                }
                gradients.push_back(layer_gradients);
            }
            //Update model with adding new gradients
            for (int i_layer = 0; i_layer < layer_amount; i_layer++) {
                int node_amount = (*ptr_layer)[i_layer].size();
                for (int i_node = 0; i_node < node_amount; i_node++) {
                    int weights_amount = (*ptr_layer)[i_layer][i_node].get_size();
                    for (int i_weight = 0; i_weight < weights_amount; i_weight++) {
                        double gradient = (gradients[i_layer][i_node].weight_gradients).at(i_weight);
                        (*ptr_layer)[i_layer][i_node].add(gradient, i_weight);
                    }
                    double gradient = gradients[i_layer][i_node].bias_gradient;
                    (*ptr_layer)[i_layer][i_node].add(gradient, -1);
                }
            }
        }
        //Data collection
        avg_losses.push_back(average_loss(temp_losses));
        temp_losses.clear();
    }
    int input_amount = ptr_Main_model->length_inputs_needed();
    //Return Data and Model in python format
    cout << "["; //(1
    //Data
    cout << "["; //(2.1
    for (int i = 0; i < avg_losses.size()-1; i++){
        cout << avg_losses.at(i) << ", ";
    }
    cout << avg_losses.at(avg_losses.size() - 1);
    cout << "], "; //2.1)
    //Model
    //Info
    cout << "["; //(2.2
    cout << input_amount << ", ";
    for (int i = 0; i < layer_amount-1; i++) {
        cout << (*ptr_layer)[i].size() << ", ";
    }
    cout << (*ptr_layer)[layer_amount-1].size();
    cout << "], "; //2.2)
    //Parameters
    cout << "["; //(2.3
    //Compatibility reasons add pm for "input" layers even though it's not needed.
    cout << "["; //(3.1
    for (int i = 0; i < input_amount; i++) {
        if (i == input_amount - 1){
            cout << "[0]";
        }
        else {
            cout << "[0], ";
        }
    }
    cout << "], "; //3.1)
    for (int i_layer = 0; i_layer < layer_amount; i_layer++) {
        cout << "["; //(3.2+
        int node_amount = (*ptr_layer)[i_layer].size();
        for (int i_node = 0; i_node < node_amount; i_node++) {
            cout << "["; //(4+
            int weights_amount = (*ptr_layer)[i_layer][i_node].get_size();
            for (int i_weight = 0; i_weight < weights_amount; i_weight++) {
                cout << (*ptr_layer)[i_layer][i_node].get_weights(i_weight) << " ,";
            }
            cout << (*ptr_layer)[i_layer][i_node].get_bias();
            if (i_node == node_amount - 1) {
                cout << "]";
            }
            else{ 
                cout << "], "; 
            } //4+)
        }
        if (i_layer == layer_amount - 1) {
            cout << "]"; //3.2+)
        }
        else {
            cout << "], ";
        }
    }
    cout << "]"; //2.3)
    cout << "]"; //1)
}

int main()
{
    using std::cout;
    int io_amount = 0; 
    int length_input; int length_output;
    vector<vector<double>*>* ptr_data_inputs = new vector<vector<double>*>; 
    vector<vector<double>*>* ptr_data_outputs = new vector<vector<double>*>;
    int epochs;
    string line;
    
    //Define Model pointer
    Model Main_Model = Model(get_amount_layers());
    Model* ptr_Main_Model = &Main_Model;

    //Lengths (node input and output)
    length_input = ptr_Main_Model->length_inputs_needed();
    length_output = ptr_Main_Model->length_outputs_expected();

    //Inputs
    char choice;
    while (getline(cin, line)) {
        stringstream ss(line);
        ss >> choice;
        switch (choice) {
            case 'p': //Set up parameters per node
                Load_Data_P(ptr_Main_Model);
                break;
            case 'd': //Get info
                if (ss >> io_amount) {/*io_amount read correctly.*/ }
                else {cout << "Invalid io_amount input" << std::endl;}
                break;
            case 'i': //Get inputs
                fill_io_pointers(ptr_data_inputs, io_amount, length_input);
                break;
            case 'o': //Get outputs
                fill_io_pointers(ptr_data_outputs, io_amount, length_output);
                break;
            case 't': //Train
                if (ss >> epochs) {/*epochs read correctly*/}
                else{cout << "Invalid epochs input"; }
                train(ptr_Main_Model, ptr_data_inputs, ptr_data_outputs, epochs, io_amount);
                break;
            default: return -1;
            }
        if (choice == 'b') {
            break;
        }
    }
    
    /*First Forward Pass
    vector<double> Output = ptr_Main_Model->Forward_pass(vector<double> {1, 2});
    for (int i = 0; i < Output.size(); i++) {
        cout << Output.at(i) << "\n";
    }
    */
  
    cout.flush();
    //Clean
    ptr_Main_Model->finish();
    delete[] ptr_Main_Model;
    if (ptr_data_inputs != nullptr && ptr_data_outputs != nullptr) {
        for (int i = 0; i < io_amount; i++) {
            delete[] (*ptr_data_inputs)[i];
            delete[] (*ptr_data_outputs)[i];
        }
        delete[] ptr_data_inputs;
        delete[] ptr_data_outputs;
    }
    return 0;
}
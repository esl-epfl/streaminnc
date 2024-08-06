# C++ Implementation of StreamiNNC for accelerometry-based seizure detection

The StreamiNNC implementation in C++ for the acceleration-based seizure CNN is located in ```./streaminnc``` and the approximate streaming in ```./streaminnc_approximate```. 

To compile and run, run the following commands:

1. Go to the implementation folder (e.g. ```./streaminnc```):
```
cd ./streaminnc
```

2. Run ```make```:
```
make
```
The executable is placed inside ```./bin``` and prints the execution time after running. 

Before compiling and running the CNN weights have to be compiled into a header file using [output_model_as_header](../output_model_as_header.py) python tool. This outputs two header files:
1. One containing the model weights (```../saved_models/decomposed_layers/weight_definitions.h```)
2. And one containing an example input(```test_data.h```). 

To set the overlap modify the following parameters:
1. ```window_stride``` in ```output_model_as_header.py``` 
2. ```const int N_samples``` in ```main.cpp``` and 
3. ```const int N_time_samples``` in ```main.cpp```

Both headers have to be placed in ```./src```. 
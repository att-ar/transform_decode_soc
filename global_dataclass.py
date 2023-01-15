from dataclasses import dataclass

@dataclass
class G:
    #preprocess
    capacity = 18.02 # battery capacity in Ampere hours
    window_time = 96 #seconds
    window_size = 32
    slicing = window_time // window_size
    batch_size = 16
    
    tgt_len = 5 # decoder input sequence length, same length as tranformer output
    src_len = window_size - tgt_len # encoder input sequence length, the 5 is an arbitrary number
    
    #network
    dense_dim = 32
    model_dim = 128
    num_features = 3 # current, voltage, and soc at t minus G.window_size -> t minus 1
    num_heads = 16
    num_layers = 6
    #learning_rate_scheduler
    T_i = 1
    T_mult = 2
    T_cur = 0.0
    #training
    epochs = 129 #should be T_i + a power of T_mult, ex) T_mult = 2 -> epochs = 2**5 + 1 = 32+1 = 33
    learning_rate = 0.0045
    min_learning_rate = 7e-11
#     weight_decay = 0.0 #No weight decay param in the the keras optimizers
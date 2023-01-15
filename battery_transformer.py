import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention, Dense, Input, Dropout, BatchNormalization
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from global_dataclass import G
from transformer_helper_dc import *

#Dense Layers
def FullyConnected():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(G.dense_dim, activation='relu',
                              kernel_initializer = tf.keras.initializers.HeNormal(),
                              bias_initializer = tf.keras.initializers.RandomUniform(minval=0.005, maxval = 0.08)
                             ),
        # (G.batch_size, G.window_size, G.dense_dim)
        tf.keras.layers.BatchNormalization(momentum = 0.98, epsilon=5e-4),
        tf.keras.layers.Dense(G.dense_dim, activation='relu',
                              kernel_initializer = tf.keras.initializers.HeNormal(),
                              bias_initializer = tf.keras.initializers.RandomUniform(minval=0.001, maxval = 0.01)
                             ),
        # (G.batch_size, G.window_size, G.dense_dim)
        tf.keras.layers.BatchNormalization(momentum = 0.95, epsilon=5e-4)
    ])


#Encoder Layer
class EncoderLayer(tf.keras.layers.Layer):
    """
    The encoder layer is composed by a multi-head self-attention mechanism,
    followed by a simple, positionwise fully connected feed-forward network. 
    This archirecture includes a residual connection around each of the two 
    sub-layers, followed by batch normalization.
    """
    def __init__(self,
                 num_heads,
                 num_features,
                 dense_dim,
                 dropout_rate,
                 batchnorm_eps):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(
            num_heads = num_heads,
            key_dim = dense_dim,
            dropout = dropout_rate,
            kernel_initializer = tf.keras.initializers.HeNormal(),
            kernel_regularizer = tf.keras.regularizers.L2(1e-4),
            bias_initializer = tf.keras.initializers.RandomUniform(minval=0.001, maxval = 0.01)
                                     )
        
        #feed-forward-network
        self.ffn = FullyConnected()
        
        
        self.batchnorm1 = BatchNormalization(momentum = 0.95, epsilon=batchnorm_eps)
        self.batchnorm2 = BatchNormalization(momentum = 0.95, epsilon=batchnorm_eps)

        self.dropout_ffn = Dropout(dropout_rate)
    
    def call(self, x, training):
        """
        Forward pass for the Encoder Layer
        
        Arguments:
            x -- Tensor of shape (G.batch_size, G.window_size, G.num_features)
            training -- Boolean, set to true to activate
                        the training mode for dropout layers
        Returns:
            encoder_layer_out -- Tensor of shape (G.batch_size, G.window_size, G.num_features)
        """
        # Dropout is added by Keras automatically if the dropout parameter is non-zero during training
        
        attn_output = self.mha(query = x,
                               value = x) # Self attention
        
        out1 = self.batchnorm1(tf.add(x, attn_output))  # (G.batch_size, G.src_len, G.dense_dim)
        
        ffn_output = self.ffn(out1)
    
        ffn_output = self.dropout_ffn(ffn_output) # (G.batch_size, G.src_len, G.dense_dim)
        
        encoder_layer_out = self.batchnorm2(tf.add(ffn_output, out1))
        # (G.batch_size, G.src_len, G.dense_dim)
        return encoder_layer_out

#Encoder itself
class Encoder(tf.keras.layers.Layer):
    """
    The entire Encoder starts by passing the input to an embedding layer 
    and using positional encoding to then pass the output through a stack of
    encoder Layers
        
    """  
    def __init__(self,
                 num_layers = G.num_layers,
                 num_heads = G.num_heads,
                 num_features = G.num_features,
                 dense_dim = G.dense_dim,
                 maximum_position_encoding = G.src_len,
                 dropout_rate=0.15,
                 batchnorm_eps=1e-4):
        
        super(Encoder, self).__init__()

        self.num_layers = num_layers

        #linear input layer
        self.lin_input = tf.keras.layers.Dense(dense_dim, activation="relu")
        
        self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                                dense_dim)


        self.enc_layers = [EncoderLayer(num_heads = num_heads,
                                        num_features = num_features,
                                        dense_dim = dense_dim,
                                        dropout_rate = dropout_rate,
                                        batchnorm_eps = batchnorm_eps) 
                           for _ in range(self.num_layers)]
        
    def call(self, x, training):
        """
        Forward pass for the Encoder
        
        Arguments:
            x -- Tensor of shape (G.batch_size, G.window_size, G.num_features)
            training -- Boolean, set to true to activate
                        the training mode for dropout layers
            mask -- Boolean mask to ensure that the padding is not 
                    treated as part of the input
        Returns:
            Tensor of shape (G.batch_size, G.window_size, G.dense_dim)
        """
        x = self.lin_input(x)
        seq_len = tf.shape(x)[1]
        x += self.pos_encoding[:, :seq_len, :]
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training)
            
        return x # (G.batch_size, G.src_len, G.dense_dim)

#Decoder Layer

class DecoderLayer(tf.keras.layers.Layer):
    """
    The decoder layer is composed by two multi-head attention blocks, 
    one that takes the new input and uses self-attention, and the other 
    one that combines it with the output of the encoder, followed by a
    fully connected block. 
    """
    def __init__(self,
                 num_heads,
                 num_features,
                 dense_dim,
                 dropout_rate,
                 batchnorm_eps):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(
            num_heads = num_heads,
            key_dim = dense_dim,
            dropout = dropout_rate,
            kernel_initializer = tf.keras.initializers.HeNormal(),
            kernel_regularizer = tf.keras.regularizers.L2(1e-4),
            bias_initializer = tf.keras.initializers.RandomUniform(minval=0.001, maxval = 0.01)
                                     )

        self.mha2 = MultiHeadAttention(
            num_heads = num_heads,
            key_dim = dense_dim,
            dropout = dropout_rate,
            kernel_initializer = tf.keras.initializers.HeNormal(),
            kernel_regularizer = tf.keras.regularizers.L2(1e-4),
            bias_initializer = tf.keras.initializers.RandomUniform(minval=0.001, maxval = 0.01)
                                     )

        self.ffn = FullyConnected()

        self.batchnorm1 = BatchNormalization(momentum = 0.95, epsilon=batchnorm_eps)
        self.batchnorm2 = BatchNormalization(momentum = 0.95, epsilon=batchnorm_eps)
        self.batchnorm3 = BatchNormalization(momentum = 0.95, epsilon=batchnorm_eps)

        self.dropout_ffn = Dropout(dropout_rate)
    
    def call(self, y, enc_output, dec_ahead_mask, enc_memory_mask, training):
        """
        Forward pass for the Decoder Layer
        
        Arguments:
            y -- Tensor of shape (G.batch_size, G.tgt_len, 1) #the soc values for the batches
            enc_output --  Tensor of shape(G.batch_size, G.num_features)
            training -- Boolean, set to true to activate
                        the training mode for dropout and batchnorm layers
        Returns:
            out3 -- Tensor of shape (G.batch_size, G.tgt_len, 1)
        """
        
        # BLOCK 1
        # Dropout will be applied during training only
        mult_attn_out1 = self.mha1(query = y,
                                   value = y,
                                   attention_mask = dec_ahead_mask,
                                   return_attention_scores=False)
        # (G.batch_size, G.tgt_len, G.dense_dim)
        
        Q1 = self.batchnorm1(tf.add(y,mult_attn_out1))

        # BLOCK 2
        # calculate self-attention using the Q from the first block and K and V from the encoder output. 
        # Dropout will be applied during training
        mult_attn_out2 = self.mha2(query = Q1,
                                   value = enc_output,
                                   key = enc_output,
                                   attention_mask = enc_memory_mask,
                                   return_attention_scores=False)
        
        mult_attn_out2 = self.batchnorm2( tf.add(mult_attn_out1, mult_attn_out2) )
                
        #BLOCK 3
        # pass the output of the second block through a ffn
        ffn_output = self.ffn(mult_attn_out2)
        
        # apply a dropout layer to the ffn output
        ffn_output = self.dropout_ffn(ffn_output)
        
        out3 = self.batchnorm3( tf.add(ffn_output, mult_attn_out2) )
        return out3
    

#Decoder itself

class Decoder(tf.keras.layers.Layer):
    """
    
    """ 
    def __init__(self,
                 num_layers = G.num_layers,
                 num_heads = G.num_heads,
                 num_features = G.num_features,
                 dense_dim = G.dense_dim,
                 target_size = G.num_features,
                 maximum_position_encoding = G.tgt_len,
                 dropout_rate=0.15,
                 batchnorm_eps=1e-5):
        super(Decoder, self).__init__()
        
        self.num_layers = num_layers
        self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                                dense_dim)

        #linear input layer
        self.lin_input = tf.keras.layers.Dense(dense_dim, activation="relu")
        
        self.dec_layers = [DecoderLayer(num_heads,
                                        num_features,
                                        dense_dim,
                                        dropout_rate,
                                        batchnorm_eps
                                       ) 
                           for _ in range(self.num_layers)]
        #look_ahead_masks for decoder:
        self.dec_ahead_mask = create_look_ahead_mask(G.tgt_len, G.tgt_len)
        self.enc_memory_mask = create_look_ahead_mask(G.tgt_len, G.src_len)
    
    def call(self, y, enc_output, training):
        """
        Forward  pass for the Decoder
        
        Arguments:
            y -- Tensor of shape (G.batch_size, G.tgt_len, G.dense_dim) #the final SOC values in the batches
            enc_output --  Tensor of shape(G.batch_size, G.src_len, G.dense_dim)
            training -- Boolean, set to true to activate
                        the training mode for dropout layers
        Returns:
            y -- Tensor of shape (G.batch_size, G.tgt_len, 1)
        """
        y = self.lin_input(y) #maps to dense_dim, the dimension of all the sublayer outputs.
        
        seq_len = tf.shape(y)[1]
        
        y += self.pos_encoding[:, :seq_len, :]

        # use a for loop to pass y through a stack of decoder layers and update attention_weights
        for i in range(self.num_layers):
            # pass y and the encoder output through a stack of decoder layers and save attention weights
            y = self.dec_layers[i](y,
                                   enc_output,
                                   self.dec_ahead_mask,
                                   self.enc_memory_mask,
                                   training)
            
        return y


#Transformer

class Transformer(tf.keras.Model):
    """
    Complete transformer with an Encoder and a Decoder
    """
    def __init__(self,
                 num_layers = G.num_layers,
                 num_heads = G.num_heads,
                 dense_dim = G.dense_dim,
                 src_len = G.src_len,
                 tgt_len = G.tgt_len,
                 max_positional_encoding_input = G.src_len,
                 max_positional_encoding_target = G.tgt_len):
        super(Transformer, self).__init__()

        self.tgt_len = tgt_len
        self.src_len = src_len
        
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.linear_map = tf.keras.Sequential([
            tf.keras.layers.Dense(
                dense_dim, activation = "relu",
                kernel_initializer = tf.keras.initializers.HeNormal(),
                bias_initializer = tf.keras.initializers.RandomUniform(minval=0.001, maxval = 0.02)
                                  ),
            tf.keras.layers.BatchNormalization(momentum = 0.97, epsilon=5e-4),

            tf.keras.layers.Dense(
                1, activation = "sigmoid",
                bias_initializer = tf.keras.initializers.RandomUniform(minval=0.001, maxval = 0.005)
                                 )
                                              ])
    
    def call(self, x, training):
        """
        Forward pass for the entire Transformer
        Arguments:
            x -- Tensor of shape (G.batch_size, G.window_size, G.num_features)
                 An array of the windowed voltage, current and soc data
            training -- Boolean, set to true to activate
                        the training mode for dropout and batchnorm layers
        Returns:
            final_output -- SOC prediction at time t
        
        """
        enc_input = x[:, :self.src_len, :]
        dec_input = x[:, -self.tgt_len:, -1:] #only want the SOC thats why -1 is there
        
        enc_output = self.encoder(enc_input, training) # (G.batch_size, G.src_len, G.num_features)
        
        dec_output = self.decoder(dec_input, enc_output, training)
        # (G.batch_size, G.tgt_len, 1)

        final_output = self.linear_map(dec_output) # (G.batch_size, G.tgt_len, 1)

        return final_output


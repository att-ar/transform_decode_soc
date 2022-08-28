# Transformers + TensorFlow and Pandas for SOC Estimation

**The Testing branch is the most up to date**

Building a transformer neural network using TensorFlow and Transformers in Python with the goal of prediciting Li-ion State of Charge based on real time voltage, current and delta time data.

The transformer network uses Batch Normalization instead of the Layer Normalization typically found in NLP.
This was done because literature said it proved significantly more effective than the NLP application of transformers.

The transformer's input will be voltage, current, and previous SOC points in a batch of windowed data of shape:<br>
`(G.batch_size, G.window_size, G.num_features)`

The voltage, current and soc data will be from time: $$t - \text{windowsize} - 1 \rightarrow t - 1$$<br>

- The encoder's input will be from time: $t - \text{windowsize} - 1 \rightarrow t - \text{targetlength}$;<br>
- The decoder's input will be from time: $t - 1 - \text{targetlength} \rightarrow t - 1$;<br>
- The decoder's output should be from time: $t - \text{targetlength} \rightarrow t$;
- The transformer output should be the decoder output with shape `(G.batch_size, G.tgt_len, 1)`;
<br>
Note that the value that is actually wanted is the one at time $t$

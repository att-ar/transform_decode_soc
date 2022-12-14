import tensorflow as tf
import numpy as np
from os import environ
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def get_angles(pos, k, d: int):
    """
    Get angles to be used in the positional encoding vectors

    Arguments:
        pos -- Column vector containing the positions [[0], [1], ...,[N-1]]
        k --   Row vector containing the dimension span [[0, 1, 2, ..., d-1]]
        d -- Encoding size

    Returns:
        angles -- (pos, d) np.array
    """
    # Get i from dimension span k
    i = k // 2
    # Calculate the angles using pos, i and d
    angles = pos / (10000 ** (2 * i / d))

    return angles


def positional_encoding(positions: int, d: int):
    """
    Precomputes a matrix with all the positional encodings

    Arguments:
        positions - Maximum number of positions to be encoded
        d - Encoding size

    Returns:
        pos_encoding - (1, position, d_model) matrix with the positional encodings
    """
    angle_rads = get_angles(np.arange(positions)[:, np.newaxis],
                            np.arange(d)[np.newaxis, :],
                            d)

    # apply sin to even indices 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, :, :].reshape(1, positions, d)

    # casts tensor to float dtype
    return tf.cast(pos_encoding, dtype=tf.float32)


def create_look_ahead_mask(dim1, dim2):
    """
    Returns an upper triangular matrix filled with ones.
    Lets the training model check if it got predictions right by having access to the actual output

    Arguments:
        sequence_length -- matrix size (sequence length is the number of time steps per input
                           input.shape = [batch_size, sequence_length, num_features])

    Returns:
        mask -- (size, size) tensor

    >>>create_look_ahead_mask(5)
    <tf.Tensor: shape=(1, 5, 5), dtype=float32, numpy=
    array([[[-0.e+00, -1.e+11, -1.e+11, -1.e+11, -1.e+11],
            [-0.e+00, -0.e+00, -1.e+11, -1.e+11, -1.e+11],
            [-0.e+00, -0.e+00, -0.e+00, -1.e+11, -1.e+11],
            [-0.e+00, -0.e+00, -0.e+00, -0.e+00, -1.e+11],
            [-0.e+00, -0.e+00, -0.e+00, -0.e+00, -0.e+00]]], dtype=float32)>
    """
    mask = (1 - tf.linalg.band_part( tf.ones((dim1,dim2)), -1, 0) ) * -1e11
    return mask

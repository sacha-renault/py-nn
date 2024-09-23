def initializer_xavier_relu(input_features, output_features) -> float:
    return (2 / input_features) ** (1 / 2)

def initializer_xavier_tanh(input_features, output_features) -> float:
    return (1 / input_features) ** (1 / 2)
from .. import xp

def initializer_xavier_relu(input_features, output_features) -> float:
    stddev = (2 / input_features) ** (1 / 2)
    return xp.random.normal(0, stddev, (input_features + output_features))

def initializer_xavier_tanh(input_features, output_features) -> float:
    stddev = (1 / input_features) ** (1 / 2)
    return xp.random.normal(0, stddev, (input_features + output_features))

def he_initializer(input_features, output_features):
    limit = (6.0 / (input_features + output_features)) ** 0.5
    return xp.random.rand(input_features, output_features) * limit * 2 - limit 
import pandas as pd
from io import StringIO


# Read and encode file
def encodeFile(input, output):
    # Read file
    data = pd.read_csv(input)

    # Class column 'e' edible = 0, 'p' poisonous = 1
    data['class'] = data['class'].map({'e' : 0, 'p' : 1})

    # One hot encode data
    encoded = pd.get_dummies(data)

    print('Number of Columns: {} -> {}'.format(len(data.columns), len(encoded.columns)))

    # Save to csv file
    if output is not None:
        encoded.to_csv(output, index=False)

    return encoded


# Read and encode line
def encodeLine(input):
    # Prepare input for reading
    input = StringIO(input)

    # Read input
    data = pd.read_csv(input, header=None)

    # One hot encode data
    encoded = pd.get_dummies(data)

    return encoded

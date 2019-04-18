import os
import matplotlib.pyplot as plt

data_path = "test/data"
train_path = os.path.join(data_path, "ptb.train.txt")


def read_data(file_path):
    with open(file_path) as f:
        return f.read().replace("\n", "<eos>\n").split("\n")

def sentence_length(data):
    data = map(lambda x:x.split(), data)
    data = map(lambda x:len(x), data)
    data = list(data)
    return data




data = read_data(train_path)
data = sentence_length(data)

print(len(data))
plt.grid(True)
plt.hist(data, bins=20,  facecolor='g')
plt.show()
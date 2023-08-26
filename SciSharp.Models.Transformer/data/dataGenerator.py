from tensorflow import keras
vocab_size = 20000  # Only consider the top 20k words
maxlen = 200  # Only consider the first 200 words of each movie review
(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=vocab_size)
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")

with open("imdb_train.txt",'w') as f:
    for i in range(len(y_train)):
        f.write(str(y_train[i])+str(list(x_train[i]))+"\n")
with open("imdb_test.txt",'w') as f:
    for i in range(len(y_val)):
        f.write(str(y_val[i])+str(list(x_val[i]))+"\n")

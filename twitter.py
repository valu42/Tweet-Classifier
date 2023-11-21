import numpy as np

import pandas as pd

from jax import random
from jax import vmap
from jax import value_and_grad
from jax import jit 

from jax.nn import swish
from jax.nn import logsumexp
from jax.nn import one_hot

import jax.numpy as jnp

import time

import random as rng

from collections import Counter

df = pd.read_csv('twitter_data/tweets.csv').dropna()

data = list(zip(df['clean_text'], df['category']))

key = random.PRNGKey(0)

def purify(tweet):
    return ''.join([i for i in tweet if i.isalpha() or i == ' '])

# Shuffle data using jnp.random
rng.seed(0)
rng.shuffle(data)

train_data = data[:8000]
test_data = data[8000:10000]

NUM_LABELS = 3

def construct_dictionary(data):
    # Constructs a dictionary of words that appear at least 25 times in the training dataset
    word_count = Counter()

    for tweet, _ in data:
        words = purify(tweet).split()
        word_count.update(words)
    
    filtered_words = [word for word in word_count if word_count[word] >= 25]
    dictionary = {word: index for index, word in enumerate(filtered_words)}

    return dictionary

dictionary = construct_dictionary(train_data)

unknown_word_index = len(dictionary)
x_length = len(dictionary) + 1

def encode_data(data, dictionary):
    # Encodes the data into one-hot vectors
    encoded_tweets = []
    labels = []

    for tweet, label in data:
        word_indexes = [dictionary.get(word, unknown_word_index) for word in purify(tweet).split()]

        one_hot_tweet = np.zeros(x_length)
        one_hot_tweet[word_indexes] = 1

        encoded_tweets.append(one_hot_tweet)
        labels.append(one_hot(label + 1, NUM_LABELS))

    return encoded_tweets, labels

def batch_data(data, batch_size):
    x_length = len(data[0]) - len(data[0]) % batch_size
    y_length = len(data[1]) - len(data[1]) % batch_size

    x_data_trimmed = data[0][:x_length]
    y_data_trimmed = data[1][:y_length]

    x_batches = [x_data_trimmed[i:i + batch_size] for i in range(0, x_length, batch_size)]
    y_batches = [y_data_trimmed[i:i + batch_size] for i in range(0, y_length, batch_size)]

    return jnp.array(x_batches), jnp.array(y_batches)

train_data = encode_data(train_data, dictionary)
test_data = encode_data(test_data, dictionary)

batch_size = 32

train_data = batch_data(train_data, batch_size)
test_data = batch_data(test_data, batch_size)

LAYER_SIZES = [x_length, 512, 3]
PARAM_SCALE = 0.01

def init_network_params(sizes, key = random.PRNGKey(0), scale=1e-2):
    def random_layer_params(m, n, key, scale=1e-2):
        w_key, b_key = random.split(key)
        return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n, ))
    
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k, scale) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]
params = init_network_params(LAYER_SIZES, key, PARAM_SCALE)

def predict(params, tweet):
    activations = tweet
   
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = swish(outputs)
    
    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return logits
batched_predict = vmap(predict, in_axes=(None, 0))

def l2_regularization(params, lambda_):
    regularization_term = 0

    for param in params:
        regularization_term += jnp.sum(jnp.square(param[0]))
        regularization_term += jnp.sum(jnp.square(param[1]))

    regularization_term *= lambda_

    return regularization_term

def loss(params, tweets, targets, lambda_):
    logits = batched_predict(params, tweets)
    log_preds = logits - logsumexp(logits)
    return -jnp.mean(targets * log_preds) + l2_regularization(params, lambda_)

INIT_LR = 1.0
DECAY_RATE = 0.99
DECAY_STEPS = 5

@jit
def update(params, x, y, epoch_number):
    loss_value, grads = value_and_grad(loss)(params, x, y, lambda_=0.001)
    lr = INIT_LR * DECAY_RATE ** (epoch_number / DECAY_STEPS)
    return [(w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(params, grads)], loss_value

num_epochs = 10000

@jit
def batch_accuracy(params, tweets, targets):
    tweets = jnp.reshape(tweets, (len(tweets), x_length))
    predicted_class = jnp.argmax(batched_predict(params,tweets), axis=1)
    targets = jnp.argmax(targets, axis=1)
    return jnp.mean(predicted_class == targets)

def accuracy(params, data):
    accs = []
    for i in range(len(data)):
        tweets = data[0][i]
        targets = data[1][i]
        accs.append(batch_accuracy(params, tweets, targets))
    return jnp.mean(jnp.array(accs))

too_high = False


for epoch in range(num_epochs):
    start_time = time.time()
    losses = []

    for i in range(len(train_data[0])):
        x = train_data[0][i]
        y = train_data[1][i]
        x = jnp.reshape(x, (len(x), x_length))
        params, loss_value = update(params, x, y, epoch, too_high = too_high)
        losses.append(loss_value)
    epoch_time = time.time() - start_time
    
    start_time = time.time()
    train_acc = accuracy(params, train_data)
    test_acc = accuracy(params, test_data)
    eval_time = time.time() - start_time
    print("params:", params[0][0])
    print(f"Epoch {epoch} in {epoch_time:0.2f} sec")
    print(f"Eval in {eval_time:0.2f} sec")
    print(f"Training set loss {jnp.mean(jnp.array(losses))}")
    print(f"Training set accuracy {train_acc}")
    print(f"Test set accuracy {test_acc}")

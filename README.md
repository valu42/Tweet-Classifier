# Tweet-Classifier
This repository is of a practice ML project. The is was to classify the tone of a tweet (negative, neutral, positive) based on the text.

# Background
This project is to learn to implement
deep learning projects and how to use JAX. I don't have much experience with deep learning so this uses quite naive idea but it works quite okay.

# Data
The dataset is from [kaggle](https://www.kaggle.com/datasets/saurabhshahane/twitter-sentiment-dataset) and the dataset has 162980 unique tweet-sentiment pair. The tweets are strings of at
most 140 characters and the sentiment is an integer from -1 to 1 (-1: negative, 0: neutral, 1: positive).

# Methodology
The project is implemented using JAX. The program labels each word that was used enough in the training dataset and represents each tweet as a kind of one-hot encoding using the labels. 
Each word is labeled with one integer and the encoding is a array of the length len(dictionary) + 1 with all zeroes except the entries corresponding to words that are present in the tweet.
This encoding is given as input to a neural network that uses Swish as it's activation function. The network uses softmax as it's output layer and Cross-Entropy loss. 

# Problems with the encoding
The encoding was the most naive one I could think (after just representing each character as integers) and it has many downsides. The first downside is that it's very sparse
and takes quite much memory. It also doesn't encode the order of the words, only which words were present. It also doesn't encode how many times each word was present (I could check if this
would make a difference). The encoding of the ordering would be important but I don't yet know how to do that. One way would bet to just have 30 words each encoded as one-hot encodings
using the label from the dictionary but this would take too much space. I'm also not sure if that would actually be useful. There's a lot of techniques and theory about working with
text so I'll probably return to this task after learning about them.

# Results
After letting the code run for a while (maybe 20 minutes) it got a bit over 90% training accuracy and 80% validation accuracy. I'm not really interested in too precise measurements as
this is only a practice work and I want to start the next one at some point. However I do want to test a couple of tricks to get the validation accuracy higher. If I think of something that
works, I'll update the repo and add the update to the "Updates" section. 
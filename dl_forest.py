# Deep learning for Kaggle Forest cover competition
# Sammy Hasan
# Sep 2017

import tensorflow as tf
import numpy as np
import os
from six.moves import cPickle as pickle

os.system('clear')
print('Initiated.')

file_name_train = 'train.pickle'
file_name_test = 'test.pickle'

with open(file_name_train,'rb') as f:
    save = pickle.load(f)

    # np.random.shuffle(save) # shuffle data for minibatch training

    split = 12096 # 80% train 20% valid
    train_dataset = save[:split,:-1].astype(np.float32)
    train_labels = save[:split,-1]-1 # need to convert to one-hot 1..7

    valid_dataset = save[split:,:-1].astype(np.float32)
    valid_labels = save[split:,-1]-1


with open(file_name_test,'rb') as f:
    test_dataset = pickle.load(f)

print('Data load complete.')
print('Train dataset: ',train_dataset.shape)
print('Train labels: ',train_labels.shape)
print('Valid dataset: ',valid_dataset.shape)
print('Valid labels: ',valid_labels.shape)
print('Test dataset: ',test_dataset.shape)


# one-hot encoding of labels
num_labels = 7
train_labels = (np.arange(num_labels) == train_labels[:,None]).astype(np.float32)
valid_labels = (np.arange(num_labels) == valid_labels[:,None]).astype(np.float32)


####
batch_size = 64
learning_rate = 0.1

size = train_dataset.shape

num_hidden_1 = 120
num_hidden_2 = 150
num_hidden_3 = 32

hidden_architecure = [size[1],54,54,54,32,32,32,16,16,num_labels]

beta = 0.001


graph = tf.Graph()

with graph.as_default():

    # Inputs.
    tf_train_dataset = tf.placeholder(tf.float32,shape = (batch_size,size[1]))
    tf_train_labels = tf.placeholder(tf.float32,shape = (batch_size,num_labels))

    tf_train_dataset_full = tf.constant(train_dataset)
    tf_valid_dataset = tf.constant(valid_dataset)
    # tf_valid_labels = tf.constant(valid_labels)


    # Variables
    def createVariables(archi):
        W = []
        B = []
        # input to H1
        weights = tf.Variable(tf.truncated_normal([archi[0],archi[1]],stddev=np.sqrt(2.0/size[1])))
        biases = tf.Variable(tf.zeros([archi[1]]))

        W.append(weights)
        B.append(biases)

        for ix_l in range(2,len(archi)):
            weights = tf.Variable(tf.truncated_normal([archi[ix_l-1],archi[ix_l]],stddev=np.sqrt(2.0/size[1])))
            biases = tf.Variable(tf.zeros([archi[ix_l]]))
            W.append(weights)
            B.append(biases)

        return W,B

    # weights_1 = tf.Variable(tf.truncated_normal([size[1],num_hidden_1],stddev=np.sqrt(2.0/size[1])))
    # biases_1 = tf.Variable(tf.zeros([num_hidden_1]))
    #
    # weights_2 = tf.Variable(tf.truncated_normal([num_hidden_1,num_hidden_2],stddev=np.sqrt(2.0/num_hidden_1)))
    # biases_2 = tf.Variable(tf.zeros([num_hidden_2]))
    # #
    # weights_3 = tf.Variable(tf.truncated_normal([num_hidden_2,num_labels],stddev=np.sqrt(2.0/num_hidden_2)))
    # biases_3 = tf.Variable(tf.zeros([num_labels]))
    # # #
    # weights_4 = tf.Variable(tf.truncated_normal([num_hidden_3,num_labels],stddev=np.sqrt(2.0/num_hidden_3)))
    # biases_4 = tf.Variable(tf.zeros([num_labels]))


    def model(dataset,W,B):
        relu = dataset

        for ix in range(len(W)-1):
            A = tf.matmul(relu,W[ix]) + B[ix]
            relu = tf.nn.relu(A)

        A = tf.matmul(relu,W[-1]) + B[-1]

        # A1 = tf.matmul(dataset,weights_1) + biases_1
        # relu = tf.nn.relu(A1)
        # A2 = tf.matmul(relu,weights_2) + biases_2
        # relu = tf.nn.relu(A2)
        # A3 = tf.matmul(relu,weights_3) + biases_3
        # # relu = tf.nn.relu(A3)
        # # A = tf.matmul(relu,weights_4) + biases_4
        return A

    W,B = createVariables(hidden_architecure)

    logits = model(tf_train_dataset,W,B)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_labels,logits=logits)) #+ beta*tf.nn.l2_loss(weights_1) + beta*tf.nn.l2_loss(weights_2) + beta*tf.nn.l2_loss(weights_3)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    predictions = tf.nn.softmax(logits)
    # train_predictions = tf.nn.softmax()

    # validation
    valid_predictions = tf.nn.softmax(model(tf_valid_dataset,W,B))
    train_predictions = tf.nn.softmax(model(tf_train_dataset_full,W,B))



def accuracy(predictions,labels):
    return(100.0 * np.sum(np.argmax(predictions,1 ) == np.argmax(labels,1))/predictions.shape[0])

num_steps = 40001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('TF initialized.')
    for step in range(num_steps):
        offset = (step*batch_size) % (train_labels.shape[0] - batch_size)
        batch_x = train_dataset[offset:offset+batch_size]
        batch_y = train_labels[offset:offset+batch_size]

        feed_dict = {tf_train_dataset:batch_x,tf_train_labels:batch_y}
        _,l,pred = session.run([optimizer,loss,predictions],feed_dict=feed_dict)
        if step%500 == 0:
            print('Step %d  loss  %.4f  accuracy  %.1f%%    ___ valid acc: %.4f%% ___ trainFull acc: %.4f%%' %(step,l,accuracy(pred,batch_y),accuracy(valid_predictions.eval(),valid_labels),accuracy(train_predictions.eval(),train_labels) ))












# eop

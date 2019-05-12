import tensorflow as tf
import numpy as np

def convertToOneHot(vector, num_classes=None):
    """
    Converts an input 1-D vector of integers into an output
    2-D array of one-hot vectors, where an i'th input value
    of j will set a '1' in the i'th row, j'th column of the
    output array.

    Example:
        v = np.array((1, 0, 4))
        one_hot_v = convertToOneHot(v)
        print one_hot_v

        [[0 1 0 0 0]
         [1 0 0 0 0]
         [0 0 0 0 1]]
    """

    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return np.float64(result)
def squash(vector):
    # vector shape: [batch_size, num of caps, unit_dim, 1]
    eps = 1e-8
    square_factor = tf.reduce_sum(tf.square(vector), axis=2, keepdims=True)
    length_factor = tf.divide(square_factor, tf.add(1.0, square_factor))
    Normalizer_factor = tf.divide(vector, tf.sqrt(square_factor)+eps)
    vec_squashed = tf.multiply(length_factor, Normalizer_factor)  # element-wise

    return (vec_squashed)
def Normalizing(vector):
    eps = 1e-8
    square_factor = tf.reduce_sum(tf.square(vector), axis=2, keepdims=True)
    Normalizer_factor = tf.divide(vector, tf.sqrt(square_factor)+eps)
    return Normalizer_factor
def custom_fc(input_layer, output_size, scope='Linear', in_dim=None):
    shape = input_layer.shape
    w_initializer = tf.contrib.layers.xavier_initializer()

    if len(shape) > 2:
        input_layer = tf.reshape(input_layer, [-1, int(np.prod(shape[1:]))])
    shape = input_layer.shape
    with tf.variable_scope(scope):
        matrix = tf.get_variable("weight",
                                 [in_dim or shape[1], output_size],
                                 dtype=tf.float32,
                                 initializer=w_initializer)
        bias = tf.get_variable("bias", [output_size], initializer=w_initializer)
        return tf.nn.bias_add(tf.matmul(input_layer, matrix), bias)
def custom_conv2d(input_layer, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, in_dim=None,
                  padding='SAME', scope="conv2d"):
    w_initializer = tf.contrib.layers.xavier_initializer()
    with tf.variable_scope(scope):
        w = tf.get_variable('w', [k_h, k_w, in_dim or input_layer.shape[-1], output_dim],
                            initializer=w_initializer)
        conv = tf.nn.conv2d(input_layer, w,
                            strides=[1, d_h, d_w, 1], padding=padding)
        b = tf.get_variable("b", shape=output_dim, initializer=w_initializer)
        conv = tf.nn.bias_add(conv, b)
        return conv
def selection(input, idx, k):
    # input.shape=[batch_size,1152,8,1]
    u = tf.expand_dims(tf.range(tf.shape(input[0])), -1)
    u2 = tf.tile(u, [1, k])
    u3 = tf.stack([u2, idx], axis=2)

    result = tf.gather_nd(input, u3)
    return result
mnist = tf.keras.datasets.mnist
(trX, trY),(teX, teY) = mnist.load_data()

trX=np.float32(trX/255.0)
teX=np.float32(teX/255.0)
trY=convertToOneHot(trY,num_classes=10)
teY=convertToOneHot(teY,num_classes=10)


X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])
num_active_caps = 70 # K in pruning layer

with tf.variable_scope('Conv1_layer'):
    conv1 = tf.contrib.layers.conv2d(X, num_outputs=256, kernel_size=9, stride=1, padding='VALID')
    conv1=tf.nn.relu(conv1)

with tf.variable_scope('PrimaryCaps_layer'):
    num_units = 8
    num_outputs = 32
    kernel_size = 9
    stride = 2
    primary_caps= (tf.contrib.layers.conv2d(conv1, num_outputs=num_outputs*num_units, kernel_size=kernel_size, stride=stride,
                                              padding="VALID"))

    tot_dim_list=primary_caps.get_shape().as_list()
    tot_dim=tot_dim_list[1]*tot_dim_list[2]*tot_dim_list[3]
    primary_caps=tf.reshape(primary_caps,[-1,tot_dim_list[1],tot_dim_list[2],num_outputs,num_units])
    primary_caps_org=tf.reshape(primary_caps,[-1,np.int32(tot_dim/num_units),num_units])
    primary_caps=squash(tf.reshape(primary_caps,[-1,np.int32(tot_dim/num_units),num_units,1]))

num_caps=np.int32(tot_dim/num_units)

with tf.variable_scope('Pruning'):
    square_factor = tf.reduce_sum(tf.square(primary_caps_org), axis=2)
    Active_amount_list_sq=tf.divide(square_factor,1.0+square_factor)
    Active_amount_list = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(primary_caps), axis=2)),[-1,num_caps])

    ORD_idx = -tf.nn.top_k(-tf.nn.top_k(Active_amount_list, k=num_active_caps).indices, k=num_active_caps).values
    ORD_value = selection(Active_amount_list, ORD_idx, num_active_caps)

    code = tf.one_hot(ORD_idx, depth=num_caps)
    code = tf.reshape(tf.reduce_sum(code, axis=1), [-1, num_caps])

    Active_caps = selection(primary_caps, ORD_idx, num_active_caps)
    Active_caps_amount = tf.sqrt(tf.reduce_sum(tf.square(Active_caps), axis=2))


Recon_err=[]
with tf.variable_scope('Ladder'):
    for k in range(10):
        with tf.variable_scope('Weight_construction'+str(k)):
            fc1 = tf.nn.relu(custom_fc(code, num_units*16*1 , scope='l1'))
            fc1 = tf.nn.relu(custom_fc(fc1, num_units*16*2  , scope='l2'))
            fc1 = tf.nn.relu(custom_fc(fc1, num_units*16*3  , scope='l3'))

            W_mat = tf.expand_dims((custom_fc(fc1, num_units* 16 * num_active_caps, scope='l4')),axis=1)
            W_mat=tf.reshape(W_mat,[-1,num_active_caps,num_units,16])

            fc1_inv = tf.nn.relu(custom_fc(code, num_units*16*1 , scope='l1_inv'))
            fc1_inv = tf.nn.relu(custom_fc(fc1_inv, num_units*16*2 , scope='l2_inv'))
            fc1_inv = tf.nn.relu(custom_fc(fc1_inv, num_units*16*3 , scope='l3_inv'))


            W_mat_inv = tf.expand_dims((custom_fc(fc1_inv, num_units * 16 * num_active_caps, scope='l4_inv')), axis=1)
            W_mat_inv = tf.reshape(W_mat_inv, [-1, num_active_caps, num_units, 16])

            fc1_c=tf.nn.relu(custom_fc(code,num_active_caps,scope='c1'))
            fc1_c = tf.nn.relu(custom_fc(fc1_c, num_active_caps, scope='c2'))
            fc1_c = tf.nn.relu(custom_fc(fc1_c, num_active_caps, scope='c3'))

            c_vec=tf.nn.sigmoid(custom_fc(fc1_c,num_active_caps,scope='c4'))
            c_vec=tf.reshape(c_vec,[-1,num_active_caps,1,1])


            D_caps_j=tf.reduce_sum(tf.multiply(c_vec,tf.matmul(Active_caps,W_mat,transpose_a=True)),axis=1,keepdims=True)
            D_caps_j_tile=Normalizing(tf.transpose(tf.tile(D_caps_j,[1,num_active_caps,1,1]),[0,1,3,2]))

            Back_caps_j=squash(tf.matmul(W_mat_inv,D_caps_j_tile))

            Recon_err_j=tf.reshape(tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(Active_caps-Back_caps_j),axis=2,keepdims=True)),axis=1),[-1,1])

            Recon_err.append(Recon_err_j)

Recon_err=tf.concat(Recon_err,axis=1)

Err_true=tf.reduce_mean(tf.reduce_sum(tf.multiply(Y,tf.square(Recon_err)),axis=1))
Err_others=tf.reduce_mean(tf.reduce_sum(tf.multiply(1.0-Y,tf.square(tf.nn.relu(0.8-Recon_err))),axis=1))

Err=Err_true+Err_others
reg_loss=tf.reduce_mean(tf.abs(code-Active_amount_list_sq))

Total_loss=Err+(1e-4)*(reg_loss)

global_step = tf.Variable(0.0, name='global_step', trainable=False)
starter_learning_rate = 1e-3
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.9, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
Train_op = optimizer.minimize(Total_loss, global_step=global_step)

#########################################  Training ####################################################
Batchsize = 50
training_epochs = 30
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for epoch in range(training_epochs):
    print ("Training "+str(epoch+1)+"epoch finished")
    for k in range(np.int32(trX.shape[0]/Batchsize)):
        batchX=trX[k*Batchsize:(k+1)*Batchsize]
        batchY=trY[k*Batchsize:(k+1)*Batchsize]
        feed_dict={X:batchX,Y:batchY}
        sess.run((Train_op),feed_dict=feed_dict)


######################################### Testing ####################################################

correct=0
incorrect=0
Batchsize=100

for k in range(np.int32(teX.shape[0] / Batchsize)):
    batchX = teX[k * Batchsize:(k + 1) * Batchsize]
    batchY = teY[k * Batchsize:(k + 1) * Batchsize]
    sess.run((Train_op), feed_dict={X: batchX})
    Rec_err_te = sess.run(Recon_err, feed_dict={X: batchX})
    pred_label = np.argmin(Rec_err_te, axis=1)
    label = np.argmax(batchY, axis=1)
    Ind = [pred_label == label]
    Ind = sum(Ind)
    tr = np.sum(Ind)
    fa = len(Ind) - tr
    correct = correct + tr
    incorrect = incorrect + fa

print ("Accuracy:"+str(np.float32(correct/(correct+incorrect))))
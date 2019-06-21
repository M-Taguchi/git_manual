import tensorflow as tf

config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="2", # specify GPU number
        allow_growth=True
    )
)

class MLP():
    def __init__(self):

        self.x = tf.placeholder("float", [None, 5])
        self.y_ = tf.placeholder("float", [None, 10])
        
        w1 = tf.Variable(tf.random_normal([5, 20]))
        b1 = tf.Variable(tf.random_normal([20]))
        w2 = tf.Variable(tf.random_normal([20, 10]))
        b2 = tf.Variable(tf.random_normal([10]))

        y1 = tf.sigmoid(tf.matmul(self.x,w1)+b1)
        logits = tf.matmul(y1,w2)+b2
        self.pred = tf.nn.softmax(logits)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=self.y_))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        self.update_model = self.optimizer.minimize(self.loss)

        correct_pred = tf.equal(tf.argmax(self.pred,1),tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

if __name__ == "__main__":
    x_data = [[0,0,0,0,1]]
    y_data = [[1,0,0,0,0,0,0,0,0,0]]

    mlp = MLP()
    epoch = 5000
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(epoch):
            _, loss_, accuracy_, pred_ = sess.run([mlp.update_model,mlp.loss,mlp.accuracy,mlp.pred],feed_dict = {mlp.x:x_data,mlp.y_:y_data})
            #loss_test_, acc_test_ = sess.run([loss, accuracy], feed_dict={x:x_test,t:t_test})
            print("[TRAIN] loss : %f, accuracy : %f" %(loss_, accuracy_))
            #print(pred_)
            #print("[TEST loss : %f, accuracy : %f" %(loss_test_, acc_test_))
    
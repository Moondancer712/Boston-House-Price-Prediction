# linear regression
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

#读取数据
df = pd.read_csv("boston.csv")
#显示数据摘要描述信息
print (df.describe())

#数据准备
df = df.values
df = np.array(df)
#数据归一化
for i in range(12):
    df[:,i] = (df[:,i]-df[:,i].min())/(df[:,i].max()-df[:,i].min())
x_data = df[:,:12]
y_data = df[:,12]

#模型定义
x = tf.placeholder(tf.float32, [None,12], name = "X")
y = tf.placeholder(tf.float32, [None,1], name = "Y")

with tf.name_scope("Model"):
    w = tf.Variable(tf.random_normal([12,1],stddev=0.01),name="W")
    b = tf.Variable(1.0, name="b")
    
    def model(x,w,b):
        return tf.matmul(x,w)+b
    
    pred = model(x,w,b)
    
#模型训练
train_epochs = 50
learning_rate = 0.01

with tf.name_scope("LostFunction"):
    loss_function = tf.reduce_mean(tf.pow(y-pred,2))
    
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

#迭代训练
loss_list = []
for epoch in range(train_epochs):
    loss_sum = 0.0
    for xs,ys in zip(x_data,y_data):
        xs = xs.reshape(1,12)
        ys = ys.reshape(1,1)
        _,loss = sess.run([optimizer,loss_function],feed_dict={x:xs,y:ys})
        
        loss_sum = loss_sum + loss
    #打乱数据顺序
    x_data,y_data = shuffle(x_data,y_data)
    
    b_temp = b.eval(session = sess)
    w_temp = w.eval(session = sess)
    loss_average = loss_sum/len(y_data)
    loss_list.append(loss_average)
    print("epoch=",epoch+1,"loss=",loss_average,"b=",b_temp,"w=",w_temp)
plt.plot(loss_list)      
#模型应用
n = np.random.randint(len(y_data))
print("测试数据: %f" % n)
x_test = x_data[n]
x_test = x_test.reshape(1,12)
predict = sess.run(pred,feed_dict = {x:x_test})
print("预测值：%f" % predict)
target = y_data[n]
print("标签值：%f" % target)
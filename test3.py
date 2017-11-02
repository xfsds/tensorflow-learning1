import tensorflow as tf

node1 = tf.constant(3.0, dtype=tf.float32, name="node1")
node2 = tf.constant(4.0, name="node2")
#   print(node1, node2)
node3 = tf.add(node1, node2)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run([node3]))

writer.close()






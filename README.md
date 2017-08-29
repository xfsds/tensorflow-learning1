# tensorflow-learning1

learning notes of github & python & tensorflow

## github operations

- **how to get github clone of a project**

    `git clone https://github.com/xxx ./path`


- **test git github operation**

    `git add` .

    `git commit -m` "commits about modification" -a

    `git push origin` master:[remote Branch Name]

## tensorflow operations


- **how to run tensorflow**

    `source ~/tensorflow/bin/activate`

    `deactivate`


- **how to run tensorboard**

    `writer = tf.summary.FileWriter('./graphs', sess.graph)`

    `tensorboard --logdir=./graphs`
Using Tensor flow to create a 2 layered network that could predict how long to roast coffee beans

```python
tempurature = 415 # in celsius
duration = 10 # in minutes

# x equal to the vector containing both parts of what makes a good roast
x = np.array([[temperature, duration]])

# initializes a 3 unit layer of sigmoid functions
layer_1 = tf.Dense(units=3, activiation='sigmoid')
a1 = layer_1(x)

# initializes a 1 unit layer of sigmoid functions
# since this is the last layer this is the out put layer
layer_2 = tf.Dense(units=1, activation='sigmoid')
a2 = layer_2(a1)

# classify by making the cut off 0.5
if a2 >= 0.5:
	y_hat = 1
else:
	y_hat = 0
```


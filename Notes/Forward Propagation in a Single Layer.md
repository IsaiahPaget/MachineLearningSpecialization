## code style

Math version w<sub>1</sub><sup>[2]</sup>
Python version w2_1

or 

w<sub>j</sub><sup>[L]</sup> = wL_j 


## Forward Propagation with a Three Unit Layer

```python
x = np.array([200, 17])
# layer 1
# unit 1
w1_1 = np.array([1, 2])
b1_1 = np.array([-1])
z1_1 = np.dot(w1_1, x) + b1_1
a1_1 = sigmoid(z1_1)

# unit 2
w1_2 = np.array([-3, 4])
b1_2 = np.array([1])
z1_2 = np.dot(w1_2, x) + b1_2
a1_2 = sigmoid(z1_2)

# unit 3
w1_3 = np.array([5, -6])
b1_3 = np.array([2])
z1_3 = np.dot(w1_3, x) + b1_3
a1_3 = sigmoid(z1_3)

a1 = np.aray([a1_1, a1_2, a1_3])

# layer 2
# unit 1

w2_1 = np.array([2, 1])
b2_1 = np.array([7, 8])
z2_1 = np.dot(w2_1, a1) + b2_1
a2_1 = sigmoid(z2_1)

output = a2_1

# this is pretty terrible code but it illistrates how the values are propagated forward
```

```python
# better version of the code

def dense(a_in, W, b):
	units = W.shape[1]
	a_out = np.zeros(units)
	for i in range(units):
		w = W[:, i]
		z = np.dot(w, a_in) + b[i]
		a_out[i] = sigmoid(z)
	return a_out
W = np.array([
			  [1, -3, 5]
			  [2, 4, -6]
])
b = np.array([-1, 1, 2])

a_in = np.array([-2, 4])

```


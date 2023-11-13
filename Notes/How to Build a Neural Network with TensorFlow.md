Creating the coffee roasting algorithm

```python
# training data
x = np.array([
			  [200, 17],
			  [120, 5],
			  [425, 20],
			  [212, 18]
			  ])
y = np.array([1, 0, 0, 1])

model = tf.Sequential([
					   tf.Dense(units=3, activation='sigmoid'),
					   tf.Dense(units=1, activation='sigmoid')
					])
model.compile(..)
model.fit(x,y)


x_new = np.array([
			  [200, 17]
			  ])

prediction = model.predict(x_new)
```

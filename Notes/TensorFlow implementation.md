```python
# classifying hand written digits
import tensorflow as tf
from tensorflow.keras import Sequential
from tensor flow.keras import Dense

model = Sequential([
	Dense(units=25, activation='sigmoid'),
	Dense(units=15, activation='sigmoid'),
	Dense(units=1, activation='sigmoid')
])

from tensorflow.keras.losses import BinaryCrossentropy

model.compile(loss=BinaryCrossentrop())

model.fit(X, Y, epochs=100)
```
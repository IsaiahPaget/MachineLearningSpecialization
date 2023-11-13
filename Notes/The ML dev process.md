### The iterative loop

- choose a architecture -> choosing the model, data, etc.
- Train the model
- Diagnose -> bias, variance and error analysis

### Email spam classifier

```python
# supervised learning

vect_X = features_of_email
y = isSpam

# features 
# could be 10,000 possible words x[0], x[1], x[2] ... x[10000]

spam = """"
	From: cheapsales@buystuff.come
	To: john smith
	Subject: buy now
	
	Deal of the week! buy now!
"""

# the vector could be an array of [{a:0}, {an: 0}, {and: 0}, and so on]
# replace every zero with a one if the word that it's looking for is the array
mail = [0..., 1, 1, 0, 1...0]

# train the model on that
```

### Error analysis

if the algorithm misclassifies n amount of examples in your data then manually go through them and categorise them based on common traits

eg:
	pharma: 21
	Deliberate misspellings: 3
	unusual email routing: 7
	phishing: 18
	spam message in embedded image: 5

it is generally best it focus on the ones that have a high impact on the algorithm like pharma or phishing, as it will have the most affect on how well the algorithm performs

### How to add more data

data augmentation:
	in the case of classifying handwritten letters you can distort the image in way like rotation, enlarger or shrinking, warping. Effectively increasing the size of the data set

### Transfer Learning

When you don't have so much data you can use another mode that is already trained on a related thing and adapt it to what you need it to do by changing the output

1. Download a neural network that has been pretrained on a large dataset with the same input type (e.g., images, audio, text) as your application or train your own

2. further train (fine tune) the network on your own data

### Dev life cycle

1. Scope the project
2. Collect data
3. train the model -> you might need to go back to step 2
4. audit: measure fairness and bias
5. Deploy: monitor the system
		Host on a server: for use in an api
		


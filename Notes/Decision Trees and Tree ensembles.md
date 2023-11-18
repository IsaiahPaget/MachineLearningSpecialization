![[Pasted image 20231115132218.png]]
root node -> the beginning of the tree
decision nodes -> take a look at the feature and decide what path to go down
leaf nodes -> the final prediction

## create decision tree

say you had 5 cats and 5 dogs in a data set you have to decide on what feature you want to split them up, you may want to start by splitting by ear shape, then you have two groups each with an odd cat or dog out, this would mean that you can to split it up further maybe by face shape or whiskers until the classification is pure, meaning no odd cat or dog out.

1. decide what features to split by
2. when to stop splitting

### measuring purity
entropy is the fraction of in this case cats in the ten data points x/10 at the root

### How to choose on what feature to split on

![[Pasted image 20231115141123.png]]
- Entropy at the root node is 0.5 because in our data set there are 5 cats and 5 dogs
- H of p is the amount of oddness out in either side of the tree where at the root node it's 1 and if there was perfect classification it would be 0 
- the last bit is calculated as the most removed from high entropy with the amount of data in each split considered

### One hot encoding

say you want to introduce a third ear shape, this binary classification so instead of having three values you need to introduce a feature for each type of ear, like floppy, pointy and oval then you populate the data point with a 1 in the category that the animal has zeros in the other types of ears for that animal

### linear features like real numbers

you gotta make a break point less than or equal to

## Tree ensemble

use multiple trees with different decision nodes and them have them all vote on one answer, majority wins

makes your over all algorithm less sensitive to change in data - more robust

### sampling with replacement

imagine a bag with four different colour tokens in it if you were to take one token out and write down what you got then put it back into the bag then grabbed a random token again
and do the same until you have written down as many tokens as there are in the bag you
would end up with another training set that is similar to what is exactly in the bag but not 
exactly the same.

you use these new training sets to make more trees that may be slightly different so that
you can make a tree ensemble

### when to use decision trees
- structured data like a spread sheet or like categorical
- not great for images, audio or text

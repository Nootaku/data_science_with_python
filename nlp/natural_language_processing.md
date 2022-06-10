# Natural Language Processing

## Description

Natural Language Processing (NLP) can be a useful tool for compiling documents, giving each documents features and then compare the features of each document.

For example we can count the word occurrences in each document. This allows to apply word segmentation on the document allowing to transform the written content into a *vector word-count* also called **Bag of Words**. A bag of words is simply a features vector meaning that we can now use mathematical operations such as the cosine similarity on a linguistic entity to determine similarity.

We can also improve on the Bag of Words by adjusting word counts based on their frequency in the corpus. Or we can also use **TF-IDF (Term Frequency - Inverse Document Frequency)**.

Whoa that's a lot of terms.

**Term Frequency** is the importance of the term within that document. We denote `TF` as a function with the parameters `d` and `t` where
$$
TF(d,t) = \text{Number of occurences of the term «t» in document «d»}
$$
**Inverse Document Frequency** is the importance of the term in the corpus of documents. Here as well we denote is as a function
$$
IDF(t) = log(D / t)
$$
where `D` is the total number of documents and `t` is the number of documents with the given term.

**TF-IDF** of the term `x` within the document `y` can then be expressed as a mathematical operation as well:
$$
W_{x, y} = tf_{x, y} \times log(\frac{N}{df_x})
$$
Here, $tf_{x, y}$ is the frequency of `x` in `y`. $df_x$ is the number of documents containing `x` and `N` is the total number of documents.

## Library installation

For NLP with Python we need to install the library [nltk](https://www.nltk.org/).

```bash
# with conda
conda install nltk

# with pip
pip install nltk
```

## Resources

If ever in need we can go to the [**Machine Learning Repository** of the UCI](https://archive.ics.uci.edu/ml/index.php) to either get some resources or to test our knowledge.

## Feature Engineering

A large part of the NLP process is feature engineering. This means to extract features from raw data.

### The process

The feature engineering process is composed of the following steps:

1. Brainstroming or testing features
2. Deciding what features to create
3. Creating the featrures
4. Testing the impact of the identified features on the task
5. Improving the features if needed
6. Repeat

Typically we use the following types of engineered features: Numerical transformations (like fractions or scaling), Category encoder, Clustering, Group aggregated values or (for numerical data) Principal component analysis.

### Text Preprocessing

Since all the ML algorithms require numerical data in order to work, we will need to do some work in order to convert our text into numerical data.

There are many methods to convert a corpus of strings into a feature vector, but the simplest is the **bag of words**.

> SciKit Learn has interesting documentation about [Working with Text Data](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html) with example on how to **Tokenize**, how to **Vectorize** and how to create **Classifiers** based on Text Data.
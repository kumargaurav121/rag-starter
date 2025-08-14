from collections import Counter
import math

sentence = "AWS and GCP are cloud platforms"
query = "AWS GCP"

def tfidf_weights(texts):
    # Tokenize
    tokenized = [t.lower().split() for t in texts]
    N = len(tokenized)
    # Document frequencies
    df = Counter()
    for doc in tokenized:
        for word in set(doc):
            df[word] += 1
    
    print(df)

    # Build TF-IDF
    tfidf_list = []

    print("tokenized", tokenized)
    for doc in tokenized:
        tf = Counter(doc)
        print("tf", tf)
        vec = {}
        for word, count in tf.items():
            idf = math.log((N+1) / (df[word]+1)) + 1
            vec[word] = count * idf
        tfidf_list.append(vec)
    return tfidf_list

docs = [sentence, query]
weights = tfidf_weights(docs)
print("Sentence TF-IDF:", weights[0])
print("Query TF-IDF:", weights[1])

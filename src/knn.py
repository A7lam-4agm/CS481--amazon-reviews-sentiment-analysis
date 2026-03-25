from data_loader import load_dataset
from preprocessing import preprocess_dataset
from vocabulary import build_vocabulary
from vectorizer import vectorize_dataset
from splitter import split_dataset

K = 3 # means how many nearest neighbors we check

def euclidean_distance(vec1, vec2):                       # This compares two document vectors.

                                                          #If two reviews are similar, the distance is smaller.
                                                            #If they are very different, the distance is bigger.
    all_words = set(vec1.keys()).union(set(vec2.keys()))
    total = 0

    for word in all_words:
        v1 = vec1.get(word, 0)
        v2 = vec2.get(word, 0)
        total += (v1 - v2) ** 2

    return total ** 0.5 


def predict_knn(train_df, test_vector, k=K):             # this will compares the test review with every training review
    distances = []
                                                         # 
    for i in range(len(train_df)):                       #
        train_vector = train_df["Vector"].iloc[i]
        train_label = train_df["Label"].iloc[i]

        dist = euclidean_distance(test_vector, train_vector)
        distances.append((dist, train_label))                 #stores the distance and label

    distances.sort(key=lambda x: x[0])                        #sorts by smallest distance

    nearest = distances[:k]                                   #keeps the nearest k

    positive_count = 0
    negative_count = 0

    for _, label in nearest:                      #counts votes
                                                  #returns the winning class
        if label == "POSITIVE":
            positive_count += 1
        else:
            negative_count += 1

    if positive_count > negative_count:
        return "POSITIVE"
    else:
        return "NEGATIVE"



if __name__ == "__main__":
    df = load_dataset("../data/Reviews.csv")
    df = preprocess_dataset(df)
    vocab = build_vocabulary(df)
    df = vectorize_dataset(df, vocab)

    train_df, test_df = split_dataset(df)

    sample_vector = test_df["Vector"].iloc[0]
    actual_label = test_df["Label"].iloc[0]

    predicted_label = predict_knn(train_df, sample_vector, k=3)

    print("Actual label   :", actual_label)
    print("Predicted label:", predicted_label)
    
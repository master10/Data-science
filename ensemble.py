import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from imblearn.over_sampling import SMOTE
from collections import Counter
import re
import nltk
import csv
from nltk.stem.snowball import SnowballStemmer
stemmer2 = SnowballStemmer("english", ignore_stopwords=True)
from nltk.corpus import stopwords # Import the stop word list

def extract_numbers( tags):
    tags_only = re.sub("[^0-9,]",'', tags).split(',')
    tagsList = map(int, tags_only)
    return tagsList

def getSolution(list1,list2,list3,list4):
    list1 = list(set(extract_numbers(list1)))
    list2 = list(set(extract_numbers(list2)))
    list3 = list(set(extract_numbers(list3)))
    list4 = list(set(extract_numbers(list4)))
    list1 = Counter(list1)
    list2 = Counter(list2)
    list3 = Counter(list3)
    list4 = Counter(list4)
    #adding the contents together
    total = list1 + list2 + list3 + list4
    solution = []
    for k,v in total.items():
        print(k,v)
        if( v >= 2):
            solution.append(k)
    return solution

def getSolution3(list1,list2,list4):
    list1 = list(set(extract_numbers(list1)))
    list2 = list(set(extract_numbers(list2)))
    #list3 = list(set(extract_numbers(list3)))
    list4 = list(set(extract_numbers(list4)))
    list1 = Counter(list1)
    list2 = Counter(list2)
    #list3 = Counter(list3)
    list4 = Counter(list4)
    #adding the contents together
    total = list1 + list2 + list4
    solution = []
    for k,v in total.items():
        print(k,v)
        if( v >= 1):
            solution.append(k)
    return solution


def review_to_words( raw_review ):

    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text()
    #
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)

    # 3. Convert to lower case, split into individual words
    #words = review_text.lower().split()
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. stemming words
    stemmedWords = [stemmer2.stem(w) for w in meaningful_words]

    #finalSetOfWords = list(set(stemmedWords))

    # 7. Join the words back into one string separated by space,
    # and return the result.
    return( " ".join(stemmedWords ))

train = pd.read_csv("/home/Documents/Personal/hackerRank/walmart/products-shelves-tagging-dataset/train.tsv", header=0,   delimiter="\t")

# Get the number of reviews based on the dataframe column size
num_reviews = train["Product Long Description"].size

# Initialize an empty list to hold the clean reviews
clean_train_reviews = []
class_label=[]
train["Product Short Description"].fillna(" ",inplace = True)
train["Product Long Description"].fillna(" ",inplace = True)
train["Seller"].fillna(" ",inplace = True)
train["Short Description"].fillna(" ",inplace = True)
train["Item Class ID"].fillna(10000,inplace = True)
train["Product Name"].fillna(" ",inplace = True)
for i in xrange( 0, num_reviews ):
    if(pd.isnull(train["tag"][i])==False):
        description = train["Product Long Description"][i] + " " \
        + train["Product Short Description"][i] + " " + \
        train["Seller"][i] + " " + \
        train["Short Description"][i] + " " + \
        train["Product Name"][i]
        review = review_to_words(description)
        review1 = review + " "+ str(train["Item Class ID"][i])
        clean_train_reviews.append(review1)
        class_label.append( train["tag"][i] )

print "Creating the bag of words...\n"

vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000,ngram_range=(1, 2),min_df = 3)

train_data_features = vectorizer.fit_transform(clean_train_reviews)

train_data_features = train_data_features.toarray()

# Initialize a Random Forest classifier with 100 trees
forest1 = RandomForestClassifier(n_estimators = 500)
forest2 = ExtraTreesClassifier(n_estimators = 500)
clf = svm.SVC(kernel='linear')
clf2 = MultinomialNB()

print "Training the random forest..."
forest1 = forest1.fit( train_data_features, class_label )

print " training extra Trees classifier"
forest2 = forest2.fit( train_data_features, class_label )

print "svm linear"
clf.fit( train_data_features, class_label )

print ("fitting naive bayes")
clf2.fit( train_data_features, class_label )

print 'reading test file'
# Read the test data
test = pd.read_csv("/home/Documents/Personal/hackerRank/walmart/products-shelves-tagging-dataset/test.tsv", header=0, delimiter="\t")


print "processing test file"
# Create an empty list and append the clean reviews one by one
num_reviews = len(test["Product Long Description"])
print "num of test cases",num_reviews
clean_test_reviews = []
class_test_label=[]
test["Product Short Description"].fillna(" ",inplace = True)
test["Product Long Description"].fillna(" ",inplace = True)
test["Seller"].fillna(" ",inplace = True)
test["Short Description"].fillna(" ",inplace = True)
test["Item Class ID"].fillna(10000,inplace = True)
test["Product Name"].fillna(" ",inplace = True)


print("generating features for test set")
for i in xrange(0,num_reviews):
    #if(pd.isnull(test["Product Long Description"][i])==False):
    description = test["Product Long Description"][i] + " " \
    + test["Product Short Description"][i] + " " + \
    test["Seller"][i] + " " + \
    test["Short Description"][i] + " " + \
    test["Product Name"][i]
    review = review_to_words(description)
    review1 = review + " "+ str(test["Item Class ID"][i])
    clean_test_reviews.append(review1)
    class_test_label.append( test["item_id"][i] )

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

result1 = forest1.predict(test_data_features)
result2 = forest2.predict(test_data_features)
result3 =  clf.predict(test_data_features)
result4 = clf2.predict(test_data_features)
commonTags = []
for i in xrange(0,num_reviews):
    final.sort()
    commonTags.append(final)
output = pd.DataFrame( data={"item_id":class_test_label, "tag":commonTags} )
output.to_csv( "tags.tsv", index=False,sep="\t")

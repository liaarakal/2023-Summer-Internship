import pandas as pd
import numpy as np
import re, string
import nltk
import pickle
import sys
import spacy

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


## Data Import

# import data using s3 buckets
df = pd.read_csv('s3://asco.nlp.sandbox/feedbackData/data.csv', encoding = "ISO-8859-1")
df.shape


# look at the counts by category
df["category 1"].value_counts()


# Remove whitespace
df['category 1'] = df['category 1'].str.strip()

# drop rows with single digit category counts
to_delete = ['Login Issues','Sign in','Generic Negative','?','Receipts','CE access','Other','Credit','Filters','networking','Meeting Reg','Improve chat','Cost','other','Abstract Book','Managing membership','Feature Request','Content','Filtering/Sorting','Logistics Issues','Survey','Meeting Networking','iplanner in app store']
df.drop(df[df['category 1'].isin(to_delete)].index, inplace=True)

df["category 1"].value_counts()

# check on Nans
df.info()

# extract only feedback and the category
df = df[['feedback','category 1']]


# ## Data Preparation

# replace empty answers with NaNs
df['feedback'].replace('-',np.nan,inplace=True)

# drop NaNs
df = df.dropna()

# drop rows with just "-"
df = df.drop(df[df[feedback] == '-'].index. inplace = True)


### Splitting the dataset into features and labels

# Features
feedback = df['feedback'].tolist()

# Labels
cat1 = df['category 1'].tolist()

len(cat1)


### Cleansing and Lemmatization

documents = []

nlp = spacy.load("en_core_web_sm")
english_stopwords = stopwords.words('english')
stemmer = WordNetLemmatizer()
for sen in range(0, len(feedback)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(feedback[sen]))
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    # Removing prefixed 'b' (not sure if necessary)
    document = re.sub(r'^b\s+', '', document)
    # Converting to Lowercase
    document = document.lower()
     # Lemmatization + remove stopwords
    document = document.split()
    document = [t for t in document if not t in english_stopwords]
    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    d = nlp(document)
    document = " ".join([word.lemma_ for word in d])
    document = document.lower()
    #print(document)
    documents.append(document)

# total reviews
len(documents)


# ### Vectorization

# initialize vectorizer
vectrz = CountVectorizer()

# run the vectorizer
feedback_matrix = vectrz.fit_transform(documents)

# extract column names
column_names = vectrz.get_feature_names_out()

# total number of columns
len(column_names)

# convert to array
feedback_array = feedback_matrix.toarray()

### Convert to a more readable form using dataframes

# set Pandas to show all columns
pd.set_option('display.max_columns', None)

# convert to the dataframe
df_feedback_postVect = pd.DataFrame(data=feedback_array,columns = column_names)

# see how many rows/columns you get total
df_feedback_postVect.shape

df_feedback_postVect.head()


# ### Sanity check to make sure the word counts are correct

# print a cleaned feedback 
print(documents[3])

# print the vectorization result and comapare if the numbers are correct
test = df_feedback_postVect.iloc[3] 
test[test != 0]


# ### Optionaly, upload to S3

# export to S3, if needed
df_feedback_postVect.to_csv('s3://asco.nlp.sandbox/feedbackData/vectorization_results.csv')


# ### Split the dataset

X_train, X_test = train_test_split(df_feedback_postVect, test_size = 0.2, random_state = 42)
Y_train, Y_test = train_test_split(cat1, test_size = 0.2, random_state = 42)


# ## Model Training

for i in range(75):
    
    i = i + 1
    ### Fitting the model ###
    # Entropy - impurity in a group of examples. Information gain is the decrease in entropy
    # GINI criterion is much faster because it is less computationally expensive.
    # ENTROPY criterion provides slightly better results because it's more computationally intensive
    
    feedback_cls = DecisionTreeClassifier(max_depth = i, criterion = "entropy", random_state=42)
    feedback_cls.fit(X_train, Y_train)

    #Predict the response for test dataset
    y_pred = feedback_cls.predict(X_test)

    # Calculate accuracy, to see how often the classifier is correct
    from sklearn import metrics
    accuracy = metrics.accuracy_score(Y_test, y_pred)*100
    print("Accuracy for " + str(i) + " iteration: %.2f%%" % accuracy)


# ## Model Tuning

# ### Try reducing the number of features using the principal component analysis (PCA)



from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# #### Standardize features by removing the mean and scaling to unit variance. This will make data looking more like normally distributed

# Initialize the scaler
sc = StandardScaler()
X = sc.fit_transform(df_feedback_postVect)


# #### Run the PCA algorithm

pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X)

print('Before PCA: ' + str(len(X[1])))
print('After PCA:' + str(len(X_reduced[1])))

X_train, X_test = train_test_split(X_reduced, test_size = 0.2, random_state = 42)

cumsum = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cumsum)

feedback_cls = DecisionTreeClassifier(max_depth = None, criterion = "entropy", random_state=42)
feedback_cls.fit(X_train, Y_train)

#Predict the response for test dataset
y_pred = feedback_cls.predict(X_test)

# Calculate accuracy, to see how often the classifier is correct
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))


# ### Try XGBoost instead

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import pprint

# before
cat1

# convert label values to numeric IDs (required by XGBoost)
le = LabelEncoder()
le.fit(cat1)
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)

# Prints the nicely formatted dictionary
pprint.pprint(le_name_mapping)

cat1 = le.fit_transform(cat1)

# after
cat1

X_train, X_test = train_test_split(df_feedback_postVect, test_size = 0.2, random_state = 42)
Y_train, Y_test = train_test_split(cat1, test_size = 0.2, random_state = 42)

# fit model no training data
model = XGBClassifier()
model.fit(X_train, Y_train)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

y_pred

predictions

# evaluate predictions
accuracy = metrics.accuracy_score(Y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

X_train, X_test = train_test_split(df_feedback_postVect, test_size = 0.2, random_state = 42)
y_train, y_test = train_test_split(cat1, test_size = 0.2, random_state = 42)

import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt

# Initialize the model and specify learning task params
xgb_clf = xgb.XGBClassifier(
                            # If you're dealing with more than 2 classes you should use softmax (or softprob).
                            # Softmax turns logits into probabilities which will sum to 1. 
                            # On basis of this, it makes the prediction which classes has the highest probabilities.
                            objective='multi:softmax',
                            # the number of classes you want to predict
                            num_class=10, 
                            # To find the optimal number of trees use an early stopping 
                            # It's a technique used to stop training when the loss on validation dataset starts to increase
                            # Specifies how many iterations we will wait for the next decrease in the loss value
                            early_stopping_rounds=10, 
                            # Evaluation metrics for validation data
                            # merror - Multiclass classification error rate. It doesn't depend upon how certain you were 
                                # about the classes you assigned, only the result.   
                            # mlogloss - Multiclass logloss (or cross-entropy loss) and penalizes models that are confident and incorrect. 
                                # It measures the performance of a classification model whose output is a probability value 
                                # between 0 and 1. 
                                # Cross-entropy loss increases as the predicted probability diverges from the actual label.
                            eval_metric=['merror','mlogloss'], 
                            # set the seed to insure consistency for every model run (can be any number)
                            seed=42)
xgb_clf.fit(X_train, 
            y_train,
            verbose=0, # set to 1 to see xgb training round intermediate results
            eval_set=[(X_train, y_train), (X_test, y_test)])

# preparing evaluation metric plots
# validation_0 dataset -> train
# validation_1 dataset -> test
results = xgb_clf.evals_result()
epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)

# xgboost 'mlogloss' plot
fig, ax = plt.subplots(figsize=(9,5))
ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
ax.legend()
plt.ylabel('mlogloss')
plt.title('GridSearchCV XGBoost mlogloss')
plt.xticks(x_axis)
plt.show()

# xgboost 'merror' plot
fig, ax = plt.subplots(figsize=(9,5))
ax.plot(x_axis, results['validation_0']['merror'], label='Train')
ax.plot(x_axis, results['validation_1']['merror'], label='Test')
ax.legend()
plt.ylabel('merror')
plt.title('GridSearchCV XGBoost merror')
plt.xticks(x_axis)

plt.show()

## ---------- Model Classification Report ----------
## get predictions and create model quality report

# extention work, not fully finished during summer of 2023

y_pred = xgb_clf.predict(X_test)

print('\n------------------ Confusion Matrix -----------------\n')
pprint.pprint(le_name_mapping)
print('\nX - Predicted, Y - Actual')
print(confusion_matrix(y_test, y_pred))

print('\n-------------------- Key Metrics --------------------')
print('\nAccuracy simply returns the percentage of labels you predicted correctly.')
print('Balanced Accuracy returns the average accuracy per class.')
print('\nAccuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
print('Balanced Accuracy: {:.2f}\n'.format(balanced_accuracy_score(y_test, y_pred)))

print('A macro-average will compute the metric independently for each class and then take \
the average (hence treating all classes equally), whereas a micro-average will aggregate \
the contributions of all classes to compute the average metric. \
In a multi-class classification setup, micro-average is preferable if you suspect there \
might be class imbalance (i.e you may have many more examples of one class than of other classes).\n')

print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))

print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))

print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))

print('\n--------------- Classification Report ---------------\n')
print('\nPrecision: What proportion of positive identifications was actually correct? When it predicts the category, it is correct XX% of the time')
print('\nRecall: What proportion of actual positives was identified correctly? It correctly identifies XX% of all labels')
print('\nF1 Score: A weighted harmonic mean of precision and recall. It combines precision and recall into a single metric. The closer to 1, the better the model')
print('\n')
print(classification_report(y_test, y_pred))
print('---------------------- XGBoost ----------------------') # unnecessary fancy styling

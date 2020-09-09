from sklearn.model_selection import KFold
from sklearn import datasets as ds
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
import time
import Algorithm_Fcs_trial
from scipy.sparse import csr_matrix
import individual_class_chi

ohsumed_collection = ds.load_files("C:/Users/Caesar/PycharmProjects/ohsumed-all", categories=['C2', 'C3', 'C7',
                                                                    'C9', 'C101', 'C105', 'C106', 'C107', 'C109', 'C202'], encoding="latin-1")
print(np.shape(ohsumed_collection.data))
#               REPLACE SPECIAL CHARACTERS WITH " "           #

count_vectorizer = CountVectorizer(encoding="latin-1", stop_words="english")

ohsumed_tokenized = count_vectorizer.fit_transform(ohsumed_collection.data)
print(ohsumed_tokenized.shape)

#ohsumed_dataframe = DataFrame.sparse.from_spmatrix(ohsumed_tokenized)
# initialize Slime Mould pop_size and iter_nr
pop_size = 10
iter_nr = 30
dimension = np.shape(ohsumed_tokenized[0][:])[1]
print("This is the dimension: ", dimension)
z = 0.03

#calculate chi value for all features/words
chi_val = individual_class_chi.chi2(ohsumed_tokenized, ohsumed_collection.target)
#chi_val = gini_index.gini_index(train_dict, my_training_data.target)
total_chi = sum(chi_val)

# Test same how many different values are in chi_val
test = chi_val[0:5,3]
print("Number of classes", len(chi_val))

#print(np.shape(chi_val)," next ", np.shape(p_val))
#print("\n", chi_val, "\n", p_val)
print("Feature selection began!")
start_time = time.time()
solution = Algorithm_Fcs_trial.algorithm_fcs(total_chi, iter_nr, pop_size, chi_val, dimension)
print("Feature selection ended in: ", (time.time()-start_time), "seconds")
# MATCH SEARCH AGENT WITH DATASET
# Remove columns from sparse matrix function
def delete_cols_csr(mat, indices):
    
    #Remove the cols denoted by ``indices`` form the CSR sparse matrix ``mat``.
    
    if not isinstance(mat, csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = list(indices)
    mask = np.ones(mat.shape[1], dtype=bool)
    mask[indices] = False
    return mat[:,mask]

# Remove columns from sparse matrix function
def delete_rows_csr(mat, indices):
    """
    Remove the cols denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = list(indices)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask,:]
# CLASSIFICATION


indices = []
for i in range(0, len(solution)):
    if solution[i] == 0:
        indices.append(i)

selected_train_dict = delete_cols_csr(ohsumed_tokenized,indices)

five_fold = KFold(n_splits=5, random_state=None, shuffle=False)
#five_fold.split(selected_train_dict)

# Variables to hold the results from the 5 fold cross-validation
RFC_acc = []
RFC_prec = []
RFC_recall = []
RFC_fscore = []
RFC_mae = []
RFC_time = []
KNN_acc = []
KNN_prec = []
KNN_recall = []
KNN_fscore = []  
KNN_mae = []
KNN_time =[]
SGD_acc = []
SGD_prec = []
SGD_recall = []
SGD_fscore = []
SGD_mae = []
SGD_time = []
N_B_acc = []
N_B_prec = []
N_B_recall = []
N_B_fscore = []
N_B_mae = []
N_B_time = []
for train_index, test_index in five_fold.split(selected_train_dict):
    #          SPLIT THE DATASET         #
    train_data, test_data = delete_rows_csr(selected_train_dict, test_index), delete_rows_csr(selected_train_dict, train_index)
    train_label, test_label = ohsumed_collection.target[train_index], ohsumed_collection.target[test_index]
    #          FIT THE NODELS            #
    # Evaluating with classifiers
    Naive_Bayes_CLS = MultinomialNB()
    start_time = time.time()
    Naive_Bayes_CLS.fit(train_data, train_label)
    predictions = Naive_Bayes_CLS.predict(test_data)
    N_B_time.append(time.time() - start_time)
    N_B_acc.append(metrics.accuracy_score(test_label, predictions))
    N_B_mae.append(metrics.mean_absolute_error(test_label, predictions))
    N_B_prec.append(metrics.precision_score(test_label, predictions, average="weighted"))
    N_B_recall.append(metrics.recall_score(test_label, predictions, average="weighted"))
    N_B_fscore.append(metrics.f1_score(test_label, predictions, average="micro"))

    SGD_CLS = SGDClassifier()
    start_time = time.time()
    SGD_CLS.fit(train_data, train_label)
    predictions = SGD_CLS.predict(test_data)
    SGD_time.append(time.time() - start_time)
    SGD_acc.append(metrics.accuracy_score(test_label, predictions))
    SGD_mae.append(metrics.mean_absolute_error(test_label, predictions))
    SGD_prec.append(metrics.precision_score(test_label, predictions, average="weighted"))
    SGD_recall.append(metrics.recall_score(test_label, predictions, average="weighted"))
    SGD_fscore.append(metrics.f1_score(test_label, predictions, average="micro"))


    KNN_CLS = KNeighborsClassifier(n_neighbors=10)
    start_time = time.time()
    KNN_CLS.fit(train_data, train_label)
    predictions = KNN_CLS.predict(test_data)
    KNN_time.append(time.time() - start_time)
    KNN_acc.append(metrics.accuracy_score(test_label, predictions))
    KNN_mae.append(metrics.mean_absolute_error(test_label, predictions))
    KNN_prec.append(metrics.precision_score(test_label, predictions, average="weighted"))
    KNN_recall.append(metrics.recall_score(test_label, predictions, average="weighted"))
    KNN_fscore.append(metrics.f1_score(test_label, predictions, average="micro"))

    RFC_CLS = RandomForestClassifier()
    start_time = time.time()
    RFC_CLS.fit(train_data, train_label)
    predictions = RFC_CLS.predict(test_data)
    RFC_time.append(time.time() - start_time)
    RFC_acc.append(metrics.accuracy_score(test_label, predictions))
    RFC_mae.append(metrics.mean_absolute_error(test_label, predictions))
    RFC_prec.append(metrics.precision_score(test_label, predictions, average="weighted"))
    RFC_recall.append(metrics.recall_score(test_label, predictions, average="weighted"))
    RFC_fscore.append(metrics.f1_score(test_label, predictions, average="micro"))

print("Average accuracy of Naive Bayes: ", np.mean(N_B_acc))
print("Average accuracy of SVM: ", np.mean(SGD_acc))
print("Average accuracy of KNN: ", np.mean(KNN_acc))
print("Average accuracy of Random Forest: ", np.mean(RFC_acc))

print("Min accuracy of Naive Bayes: ", min(N_B_acc))
print("Min accuracy of SVM: ", min(SGD_acc))
print("Min accuracy of KNN: ", min(KNN_acc))
print("Min accuracy of Random Forest: ", min(RFC_acc))

print("Max accuracy of Naive Bayes: ", max(N_B_acc))
print("Max accuracy of SVM: ", max(SGD_acc))
print("Max accuracy of KNN: ", max(KNN_acc))
print("Max accuracy of Random Forest: ", max(RFC_acc))

print("Average precision of Naive Bayes: ", np.mean(N_B_prec))
print("Average precision of SVM: ", np.mean(SGD_prec))
print("Average precision of KNN: ", np.mean(KNN_prec))
print("Average precision of Random Forest: ", np.mean(RFC_prec))

print("Average recall of Naive Bayes: ", np.mean(N_B_recall))
print("Average recall of SVM: ", np.mean(SGD_recall))
print("Average recall of KNN: ", np.mean(KNN_recall))
print("Average recall of Random Forest: ", np.mean(RFC_recall))

print("Average micro-F1 score of Naive Bayes: ", np.mean(N_B_fscore))
print("Average micro-F1 score of SVM: ", np.mean(SGD_fscore))
print("Average micro-F1 score of KNN: ", np.mean(KNN_fscore))
print("Average micro-F1 score of Random Forest: ", np.mean(RFC_fscore))

print("Average time taken Naive Bayes: %s" % np.mean(N_B_time))
print("Average time taken SVM: %s" % np.mean(SGD_time))
print("Average time taken KNN: %s" % np.mean(KNN_time))
print("Average time taken Random Forest: %s" % np.mean(RFC_time))

print("Average MAE Naive Bayes: %s" % np.mean(N_B_mae))
print("Average MAE SVM: %s" % np.mean(SGD_mae))
print("Average MAE KNN: %s" % np.mean(KNN_mae))
print("Average MAE Random Forest: %s" % np.mean(RFC_mae))

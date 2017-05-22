# Testing

###########  trying to use the kFold included - > 15 min
# from sklearn import cross_validation as cv

# X = X_train[0:5][0:5]
# y = y_train[0:5]
# k_fold = cross_validation.Kfold(len(y),10)

# def check_cross_val(X_train, y_train):
#     for k, (train, test) in enumerate(k_fold):
#         model = svm.LinearSVC(C=e)  # freezes with default SVC
#         model.fit(X_train[train], y_train)
#         acc.append(model.score(X_validation, y_validation))
#     avg_acc.append(sum(acc)/float(len(acc)))
#     best_C = C[avg_acc.index(max(avg_acc))]
    
#     return avg_acc, best_C
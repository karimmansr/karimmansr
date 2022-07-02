import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import (load_prostmat, load_data_set_ALL_AML_independent,DLBCL)
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import (SelectKBest, MutualInfoSelector,
                                       f_classif, f_regression)
from sklearn.svm import LinearSVC


def compare_methods(clf, X, y, discrete_features, discrete_target,
                    k_all=None, cv=5):
    if k_all is None:
        k_all = np.arange(1, X.shape[1] + 1)

    if discrete_target:
        t_test = SelectKBest(score_func=f_classif)
    else:
        t_test = SelectKBest(score_func=f_regression)

    max_rel = MutualInfoSelector(use_redundancy=False,
                                 n_features_to_select=np.max(k_all),
                                 discrete_features=discrete_features,
                                 discrete_target=discrete_target)

    mrmr = MutualInfoSelector(n_features_to_select=np.max(k_all),
                              discrete_features=discrete_features,
                              discrete_target=discrete_target)

    f_test.fit(X, y)
    max_rel.fit(X, y)
    mrmr.fit(X, y)

    t_test_scores = []
    max_rel_scores = []
    mrmr_scores = []

    for k in k_all:
        t_test.set_params(k=k)
        max_rel.set_params(n_features_to_select=k)
        mrmr.set_params(n_features_to_select=k)

        X_t_test = X[:, t_test.get_support()]
        X_max_rel = X[:, max_rel.get_support()]
        X_mrmr = X[:, mrmr.get_support()]

        t_test_scores.append(np.mean(cross_val_score(clf, X_f_test, y, cv=cv)))
        max_rel_scores.append(
            np.mean(cross_val_score(clf, X_max_rel, y, cv=cv)))
        mrmr_scores.append(np.mean(cross_val_score(clf, X_mrmr, y, cv=cv)))

    scores = np.vstack((t_test_scores, max_rel_scores, mrmr_scores))

    return k_all, scores


prostate = load_prostate()
X = prostate.data
y = prostate.target
k_prostate, scores_prostate = compare_methods(LinearSVC(), X, y, True, True,
                                          k_all=np.arange(1, 16))

leukemia = load_data_set_ALL_AML_independent()
X = minmax_scale(leukemia.data)
y = leukemia.target
k_leukemia, scores_leukemia = compare_methods(LinearSVC(), X, y, False, True,
                                          k_all=np.arange(1, 16))

DLBCL = load_DLBCL()
X = DLBCL.data
y = DLBCL.target
k_DLBCL, scores_DLBCL = compare_methods(RidgeCV(normalize=True), X, y,
                                              [1], False)




plt.figure(figsize=(12, 12))
plt.subplot(221)
plt.plot(k_prostate, scores_prostate[0], 'x-', label='t-test')
plt.plot(k_prostate, scores_prostate[1], 'x-', label='entropie')
plt.plot(k_prostate, scores_prostate[2], 'x-', label='mRMR')

plt.xlabel('Number of kept features')
plt.ylabel('5-fold CV average score')
plt.legend(loc='lower right')

plt.subplot(222)
plt.plot(k_leukemia, scores_leukemia[0], 'x-', label='t-test')
plt.plot(k_leukemia, scores_leukemia[1], 'x-', label='entropie')
plt.plot(k_leukemia, scores_leukemia[2], 'x-', label='mRMR')

plt.xlabel('Number of kept features')
plt.ylabel('5-fold CV average score')
plt.legend(loc='lower right')

plt.subplot(223)
plt.plot(k_DLBCL, scores_DLBCL[0], 'x-', label='t-test')
plt.plot(k_DLBCL, scores_DLBCL[1], 'x-', label='entropie')
plt.plot(k_DLBCL, scores_DLBCL[2], 'x-', label='mRMR')

plt.xlabel('Number of kept features')
plt.ylabel('5-fold CV average score')
plt.legend(loc='lower right')



plt.suptitle("Algorithm scores using different feature selection methods",
             fontsize=16)
plt.show()

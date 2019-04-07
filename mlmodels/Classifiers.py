from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score

class Classifiers(object):

    def __init__(self, featureset=None, target=None, mode='predict', path=''):
        if (mode == 'train'):
            self.__svm = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
            self.__rfc = RFC(bootstrap=True, class_weight=None, criterion='gini', max_depth=100, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)
            (self.__svm, self.__rfc) = self.__trainAll(X=list(featureset), Y=list(target))
            self.__saveModelsToFile(path)
        else:
            self.__svm = joblib.load(path + 'Mel_SVM.pkl')
            self.__rfc = joblib.load(path + 'Mel_RFC.pkl')

    def __trainAll(self, X, Y):
        return ((self.__svm).fit(X, Y), (self.__rfc).fit(X, Y))

    def __saveModelsToFile(self, path):
        joblib.dump(self.__svm, (path + 'Mel_SVM.pkl'))
        joblib.dump(self.__rfc, (path + 'Mel_RFC.pkl'))

    def predicto(self, extfeatarr, supresults):
        svm_res = (self.__svm).predict(list(extfeatarr))
        rfc_res = (self.__rfc).predict(list(extfeatarr))
        return ({
                    'SVM' : { 'Prediction Results' : svm_res, 'Accuracy' : accuracy_score(list(supresults), svm_res)},
                    'RFC': {'Prediction Results': rfc_res, 'Accuracy': accuracy_score(list(supresults), rfc_res)}
                })

    def predictForSingleImage(self, extfeatarr):
        svm_res = (self.__svm).predict(list(extfeatarr))
        rfc_res = (self.__rfc).predict(list(extfeatarr))
        return ({
                    'SVM' : { 'Prediction Results' : svm_res},
                    'RFC': {'Prediction Results': rfc_res}
                })
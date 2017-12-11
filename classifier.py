import numpy
from sklearn import preprocessing
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

def run():
    dataset = numpy.loadtxt('./train.csv', delimiter=',', skiprows=1)
    train_data = dataset[:, 1:6]
    train_labels = dataset[:, 6]

    raw_dataset = numpy.loadtxt('./__test.csv', delimiter=',', skiprows=1)
    raw_data = raw_dataset[:, 1:6]

    normalized_train_data = preprocessing.normalize(train_data)
    normalized_raw_data = preprocessing.normalize(raw_data)

    print('start training')
    model = KNeighborsClassifier(n_neighbors=10)
    model.fit(normalized_train_data, train_labels)

    print('start predict')
    expected = train_labels
    predicted = model.predict_proba(normalized_raw_data)
    # print(metrics.classification_report(expected, predicted))
    # print(metrics.confusion_matrix(expected, predicted))
    # print(model)

    print('start save')
    result = np.hstack((raw_dataset[:,0], predicted[:,:1].reshape(1, len(predicted[:,:1]))[0]))
    numpy.savetxt('result.csv', result, delimiter=',', fmt='%s,%f', comment='', header='id,prob')


if __name__ == '__main__':
    run()
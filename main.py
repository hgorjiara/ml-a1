import numpy as np
import matplotlib.pyplot as plt
import mltools as ml

def main() :
    iris = np.genfromtxt("data/iris.txt", delimiter=None)
    Y = iris[:,-1]
    X = iris[:, 0:-1]
    print X.shape
    # Part 2
    # for f in X.T:
    #     plt.hist(f)
    #     plt.show()
    # Part 3
    for f in X.T:
        print "Mean: ", np.mean(f)
        print "Standard deviation: ", np.std(f)
    # Part 4
    # pairs = [[0, 1, 4], [0, 2, 4], [0, 3, 4]]
    # colors = ['r*', 'g*', 'b*']
    # for p in pairs:
    #     for feature in iris[:, p]:
    #         plt.plot(feature[0], feature[1], colors[int(feature[2])])
    # plt.show()
    # Question 2
    # Part 1)
    # XX = X[:, [0, 1]]
    # np.random.seed(1)
    # XX, Y = ml.shuffleData(XX, Y)
    # np.random.seed(1)
    # XXtr, XXva, Ytr, Yva = ml.splitData(XX, Y, 0.75)
    # K = [1, 5, 10, 50];
    # for k in K:
    #     knn = ml.knn.knnClassify()
    #     knn.train(XXtr, Ytr, k)
    #     ml.plotClassify2D(knn, XXtr, Ytr, axis=plt)
    #     plt.title("K = ", k)
    #     plt.show()
    # Part 2
    np.random.seed(1)
    X, Y = ml.shuffleData(X, Y)
    np.random.seed(1)
    Xtr, Xva, Ytr, Yva = ml.splitData(X, Y, 0.75)
    XXtr = Xtr[:, [0,1]]
    XXva = Xva[:, [0,1]]
    K = [1, 2, 5, 10, 50, 100, 200];
    trainErr  =[]
    validErr = []
    for i,k in enumerate(K):
        knn = ml.knn.knnClassify()
        knn.train(XXtr, Ytr, k)
        YHat = knn.predict(XXtr)
        trainErr.append( np.sum(YHat != Ytr)*1.0/len(YHat) )
        YHat = knn.predict(XXva);
        validErr.append( np.sum(YHat != Yva)*1.0/len(YHat) )
        print "K = ", k, ": Error rate on training data = ", trainErr[i], ", on validation data = ", validErr[i]
    plt.semilogx(K, trainErr, color = "r", label = "Error on Training Data")
    plt.semilogx(K, validErr, color = "g", label = "Error on ")
    plt.show()

    trainErr = []
    validErr = []
    for i, k in enumerate(K):
        knn = ml.knn.knnClassify()
        knn.train(Xtr, Ytr, k)
        YHat = knn.predict(Xtr)
        trainErr.append(np.sum(YHat != Ytr) * 1.0 / len(YHat))
        YHat = knn.predict(Xva);
        validErr.append(np.sum(YHat != Yva) * 1.0 / len(YHat))
        print "K = ", k, ": Error rate on training data = ", trainErr[i], ", on validation data = ", validErr[i]
    plt.semilogx(K, trainErr, color="r", label="Error on Training Data")
    plt.semilogx(K, validErr, color="g", label="Error on Validation Data")
    plt.show()
    print "OK, I'm done."
if __name__ == '__main__':
    main()
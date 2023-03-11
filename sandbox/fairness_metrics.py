import numpy as np

def test_independence(model, X, a, b) -> bool:
    # y is the target variable
    # X is the feature matrix
    # a is the column for the first group
    # b is the column for the second group
    # for both grouqps a and b, the values are 0 or 1 (one-hot encoding)

    # create two groups
    X_a = X[X[a] == 1]
    X_b = X[X[b] == 1]

    # check prediction for group a and b
    y_pred_a = model.predict(X_a)
    y_pred_b = model.predict(X_b)

    # calculate p(y_pred=1|A=a) and p(y_pred=1|A=b)
    p_a = np.mean(y_pred_a)
    p_b = np.mean(y_pred_b)

    # print results
    print("p(y_pred=1|A={}) = ".format(a), p_a)
    print("p(y_pred=1|A={}) = ".format(b), p_b)

    proposition = (p_a == p_b)
    if proposition:
        print("The model fulfills independence")
    else:
        print("The model does not fulfill independence")
    return proposition

def test_separation(model, X, y, a, b) -> bool:
    # y is the target variable
    # X is the feature matrix
    # a is the column for the first group
    # b is the column for the second group
    # for both grouqps a and b, the values are 0 or 1 (one-hot encoding)

    # create four groups
    X_1_a = X[(X[a] == 1) & (y == 1)]
    X_1_b = X[(X[b] == 1) & (y == 1)]
    X_0_a = X[(X[a] == 1) & (y == 0)]
    X_0_b = X[(X[b] == 1) & (y == 0)]

    # check prediction all groups
    y_pred_1_a = model.predict(X_1_a)
    y_pred_1_b = model.predict(X_1_b)
    y_pred_0_a = model.predict(X_0_a)
    y_pred_0_b = model.predict(X_0_b)

    # calculate p(y_pred=1|Y=1, A=a) and p(y_pred=1|Y=1, A=b) etc
    p_1_a = np.mean(y_pred_1_a)
    p_1_b = np.mean(y_pred_1_b)
    p_0_a = np.mean(y_pred_0_a)
    p_0_b = np.mean(y_pred_0_b)

    # print results
    print("p(y_pred=1|Y=1, A={}) = ".format(a), p_1_a)
    print("p(y_pred=1|Y=1, A={}) = ".format(b), p_1_b)
    print("p(y_pred=1|Y=0, A={}) = ".format(a), p_0_a)
    print("p(y_pred=1|Y=0, A={}) = ".format(b), p_0_b)

    proposition = (p_1_a == p_1_b) & (p_0_a == p_0_b)
    if proposition:
        print("The model fulfills seperation")
    else:
        print("The model does not fulfill seperation")
    return proposition


def test_sufficiency(model, X, y, a, b) -> bool:
    X_1_a = X[(X[a] == 1) & (y == 1)]
    X_1_b = X[(X[b] == 1) & (y == 1)]
    X_0_a = X[(X[a] == 1) & (y == 0)]
    X_0_b = X[(X[b] == 1) & (y == 0)]

    y_pred_1_a = model.predict_proba(X_1_a)
    y_pred_1_b = model.predict_proba(X_1_b)
    y_pred_0_a = model.predict_proba(X_0_a)
    y_pred_0_b = model.predict_proba(X_0_b)

    # marginalize dep. variable A
    # p(y_pred=1|R=r, A=a) = p(y_pred=1|R=r, A=b)
    p_1_a = np.mean(y_pred_1_a)
    p_1_b = np.mean(y_pred_1_b)
    p_0_a = np.mean(y_pred_0_a)
    p_0_b = np.mean(y_pred_0_b)
    
    print("p(y_pred=1|Y=1, A={}) = ".format(a), p_1_a)
    print("p(y_pred=1|Y=1, A={}) = ".format(b), p_1_b)
    print("p(y_pred=1|Y=0, A={}) = ".format(a), p_0_a)
    print("p(y_pred=1|Y=0, A={}) = ".format(b), p_0_b)

    proposition = (p_1_a == p_1_b) & (p_0_a == p_0_b)
    if proposition:
        print("The model fulfills sufficiency")
    else:
        print("The model does not fulfill seperation")
    return proposition
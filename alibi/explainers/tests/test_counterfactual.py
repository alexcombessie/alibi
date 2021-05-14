import pytest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from alibi.api.defaults import DEFAULT_META_CF, DEFAULT_DATA_CF
from alibi.explainers.counterfactual import _define_func
from alibi.explainers import CounterFactual


@pytest.fixture
def logistic_iris():
    X, y = load_iris(return_X_y=True)
    lr = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=200).fit(X, y)
    return X, y, lr


@pytest.fixture
def cf_iris_explainer(request, logistic_iris):
    X, y, lr = logistic_iris
    predict_fn = lr.predict_proba
    cf_explainer = CounterFactual(predict_fn=predict_fn, shape=(1, 4),
                                  target_class=request.param, lam_init=1e-1, max_iter=1000,
                                  max_lam_steps=10)

    yield X, y, lr, cf_explainer
    keras.backend.clear_session()
    tf.keras.backend.clear_session()


@pytest.mark.parametrize('target_class', ['other', 'same', 0, 1, 2])
def test_define_func(logistic_iris, target_class):
    X, y, model = logistic_iris

    x = X[0].reshape(1, -1)
    predict_fn = model.predict_proba
    probas = predict_fn(x)
    pred_class = probas.argmax(axis=1)[0]
    pred_prob = probas[:, pred_class][0]

    func, target = _define_func(predict_fn, pred_class, target_class)

    if target_class == 'same':
        assert target == pred_class
        assert func(x) == pred_prob
    elif isinstance(target_class, int):
        assert target == target_class
        assert func(x) == probas[:, target]
    elif target_class == 'other':
        assert target == 'other'
        # highest probability different to the class of x
        ix2 = np.argsort(-probas)[:, 1]
        assert func(x) == probas[:, ix2]

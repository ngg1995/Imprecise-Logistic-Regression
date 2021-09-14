Imprecise Logistic Regression
==============================
Imprecise Logistic Regression class mimicing `scikit-learn's LogisticRegression`_ class

-----

Usage
=====
The usage is identical to scikit-learn 

.. code:: 

    from ImpLogReg import ImpLogReg
    ilr = ImpLogReg(uncertain_class = True)
    ilr.fit(data,results)

------

Documentation
=============

*class* ``ImpLogReg(uncertain_data = False, uncertain_class = False, **kwargs)``

Parameters
__________
    **uncertain_data** : *bool, default = False*
        Whether the imprecise model has interval datapoints.
    **uncertain_class**: *bool, default = False*
        Whether the imprecise model has uncertain results.
    ****kwargs**
        kwargs passed to sklearn.linear_model.LogisticRegression() function.

Attributes
__________
    **models**: *dict*
        Dictionary containing the set of LR models trained on the dataset.
    **coef\_**: *list*
        Set of possible coefficients of the features in the decision functions.
    **intercept\_**: *list*
        Set of coefficient of the features in the decision functions.
    **n\_iter\_**: *list*
        Actual number of interations for all LR models in the set 


**Methods**


+--------------------------------------------+-------------------------------------------------------------+
| `decision_function(X)`_                    | Predict confidence scores for samples.                      |
+--------------------------------------------+-------------------------------------------------------------+
| `densify()`_                               | Convert coefficient matrix to dense array format.           |
+--------------------------------------------+-------------------------------------------------------------+
| `fit(X, y, sample_weight, catagorical)`_   | Fit the model according to the given training data.         |
+--------------------------------------------+-------------------------------------------------------------+
| `get_params()`_                            | Get parameters for this estimator.                          |
+--------------------------------------------+-------------------------------------------------------------+
| `predict(X)`_                              | Predict class labels for samples in X.                      |
+--------------------------------------------+-------------------------------------------------------------+
| `predict_log_proba(X)`_                    | Predict logarithm of probability estimates.                 |
+--------------------------------------------+-------------------------------------------------------------+
| `predict_proba(X)`_                        | Probability estimates.                                      |
+--------------------------------------------+-------------------------------------------------------------+
| `score(X, y , sample_weight)`_             | Return the mean accuracy on the given test data and labels. |
+--------------------------------------------+-------------------------------------------------------------+
| `set_params(**params)`_                    | Set the parameters of this estimator.                       |
+--------------------------------------------+-------------------------------------------------------------+
| `sparsify()`_                              | Convert coefficient matrix to sparse format.                |
+--------------------------------------------+-------------------------------------------------------------+


``decision_function(X)``
------------------------

.. _decision_function(X):

Predict confidence scores for samples.

The confidence score for a sample is proportional to the signed distance of that sample to the hyperplane.

Parameters

    **X**: *array-like or sparse matrix, shape (n_samples, n_features)*
        Samples Passed to 

Returns

    *list of pba.Interval objects*
        Interval of confidence score for each sample for all *m* models

See `scikit-learn.linear_model.LogisticRegression().decision_function()`_

``densify()``
-------------

.. _densify(X):

Converts coefficient matrix to dense array format for all LR models

Returns

    *self*

See `scikit-learn.linear_model.LogisticRegression().densify()`_

``fit(X, y, sample_weight = None, catagorical = [])``
_____________________________________________________

.. _fit(X, y, sample_weight, catagorical):

Fit the model according to the given training data.

Parameters

    **X**: *{array-like, sparse matrix} of shape (n_samples, n_features)*
        Training vector, where n_samples is the number of samples and n_features is the number of features.
        If uncertain_data is true then will expect 

    **y**: *array-like of shape (n_samples,)*
        Target vector relative to X.

    **sample_weight**: *array-like of shape (n_samples,) default=None*
        Array of weights that are assigned to individual samples. If not provided, then each sample is given unit weight.

    **catagorical**: *array-like default = []*
        Array containing a list of features that contain catagorical data.

Returns

    **self** 
        Dictionary of fitted estimators


See `scikit-learn.linear_model.LogisticRegression().fit()`_

``get_params()``
----------------

.. _get_params():

Get parameters for this estimator.

Returns
    **params**: *dict*

See `scikit-learn.linear_model.LogisticRegression().get_params()`_

``predict(X)``
--------------

.. _predict(X):

Predict class labels for this estimator

Parameters

    **X**: *array-like or sparse matrix, shape (n_samples, n_features)*
        Samples

Returns

    *list of pba.Logical objects*
        Predicted class labels per sample. 0 if sample is always 0, 1 if sample is always 1 or pba.Logical(0,1) otherwise

See `scikit-learn.linear_model.LogisticRegression().predict()`_

``predict_log_proba(X)``
------------------------

.. _predict_log_proba(X):

Predict class labels for this estimator

Parameters

    **X**: *array-like or sparse matrix, shape (n_samples, n_features)*
        Samples

Returns

    *array-like of shape (n_samples, n_classes) containing pba.Interval objects*
        Returns the Interval log-probability of the sample for each class in the model, where classes are ordered as they are in self.classes\_.

See `scikit-learn.linear_model.LogisticRegression().predict_log_proba()`_


``predict_proba(X)``
---------------------

.. _predict_proba(X):

Predict class labels for this estimator

Parameters

    **X**: *array-like or sparse matrix, shape (n_samples, n_features)*
        Samples

Returns

    *array-like of shape (n_samples, n_classes) containing pba.Interval objects*
        Returns the Interval probability of the sample for each class in the model, where classes are ordered as they are in self.classes\_.

See `scikit-learn.linear_model.LogisticRegression().predict_proba()`_

``score(X, y , sample_weight)``
_______________________________

.. _`score(X, y , sample_weight)`:

Return the mean accuracy on the given test data and labels.

In multi-label classification, this is the subset accuracy which is a harsh metric since you require for each sample that each label set be correctly predicted.

Parameters

    **X**: *{array-like, sparse matrix} of shape (n_samples, n_features)*
        Training vector, where n_samples is the number of samples and n_features is the number of features.
        If uncertain_data is true then will expect 

    **y**: *array-like of shape (n_samples,)*
        Target vector relative to X.

    **sample_weight**: *array-like of shape (n_samples,) default=None*
        Array of weights that are assigned to individual samples. If not provided, then each sample is given unit weight.

Returns 

    **score**: *pba.Interval*
        Interval containing the minimum and maximum accuracy score for all models

See `scikit-learn.linear_model.LogisticRegression().score()`_

``set_params(**params)``
--------------------------

.. _`set_params(**params)`:

Parameters

    **params**: *dict*

        Estimator parameters.

Returns

    **self**: Estimators.

See `scikit-learn.linear_model.LogisticRegression().set_params()`_

``sparsify()``
--------------

.. _`sparsify()`:

Convert coefficient matrix to sparse format for each model

Returns
    **self**: Fitted estimators.

See `scikit-learn.linear_model.LogisticRegression().sparsify()`_

.. _scikit-learn's LogisticRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logisticregression#sklearn.linear_model.LogisticRegression

.. _`scikit-learn.linear_model.LogisticRegression().decision_function()`: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logisticregression#sklearn.linear_model.LogisticRegression.decision_function

.. _`scikit-learn.linear_model.LogisticRegression().densify()`: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logisticregression#sklearn.linear_model.LogisticRegression.densify

.. _`scikit-learn.linear_model.LogisticRegression().fit()`: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logisticregression#sklearn.linear_model.LogisticRegression.fit

.. _`scikit-learn.linear_model.LogisticRegression().get_params()`: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logisticregression#sklearn.linear_model.LogisticRegression.get_params

.. _`scikit-learn.linear_model.LogisticRegression().predict()`: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logisticregression#sklearn.linear_model.LogisticRegression.predict

.. _`scikit-learn.linear_model.LogisticRegression().predict_log_proba()`: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logisticregression#sklearn.linear_model.LogisticRegression.predict_log_proba

.. _`scikit-learn.linear_model.LogisticRegression().predict_proba()`: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logisticregression#sklearn.linear_model.LogisticRegression.predict_proba

.. _`scikit-learn.linear_model.LogisticRegression().score()`: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logisticregression#sklearn.linear_model.LogisticRegression.score

.. _`scikit-learn.linear_model.LogisticRegression().set_params()`: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logisticregression#sklearn.linear_model.LogisticRegression.set_params

.. _`scikit-learn.linear_model.LogisticRegression().sparsify()`: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logisticregression#sklearn.linear_model.LogisticRegression.sparsify
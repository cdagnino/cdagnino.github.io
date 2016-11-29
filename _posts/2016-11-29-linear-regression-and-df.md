---
layout: post
title: Linear regression and degrees of freedom
date:   2016-11-29 18:02:00
categories: economics, statistics, machine learning
image: 
---


In this post I go into details of how statistical learning frames the relationship between out-of-sample and within-sample fit. I also connect it with the familiar idea of degrees of freedom in linear regression, which every student of econometrics encounters.

This is the mathy cousin of the post [Econometrics and Machine Learning: objectives and comparative advantages]({% post_url 2016-11-28-econometrics-and-ML %})



<br>

**The big picture** is that, for any model

$$ \mbox{Real Prediction Error} = \mbox{within-sample prediction error} + \mbox{optimism} $$

After you fit a model with data, it's easy to calculate the within-sample prediction error, but what we really want is a measure of how our model would do when encountering new data (i.e. the *real* prediction error). To do this, we need a way to estimate how optimistic is our model. It turns out that optimism has to do with the complexity of the model:

$$ \mbox{Real Prediction Error} = \mbox{within-sample prediction error} + \mbox{model complexity} $$

In other words, if a model is very complex, the within-sample prediction record of it will be too optimistic and, hence, a bad measure of how the model would do with a new sample. 



Let's see the mathematical details. For the rest of the post, the notation will closely follow Chapter 7 of [Elements of Statistical Learning](http://statweb.stanford.edu/~tibs/ElemStatLearn/).

Let $L$ be any loss function (quadratic would do), that tells us how upset we become when our target $Y$ is far from our prediction $\hat{f}(x)$.

To fit the model, we are given a training set $T = \{(x_1, y_1), (x_2, y_2),...,(x_n, y_n)   \} $ and we obtain $\hat{f}(x)$

The true error of a model is:

$$Err_{T} = E_{X^0, Y^0} \left[L(Y^0, \hat{f} (X^0))\;|\; T\right] $$

The $0$ in $X^0, Y^0$ indicates that this is a new data point and wasn't used for training/estimating the model. The true error of a model is how good our prediction is **when we apply the model to new data**. 

Sometimes it's easier to work with the expected (true) error, which is an average over all training sets:

$$ Err = E_T E_{X^0, Y^0} \left[L(Y^0, \hat{f} (X^0))\;|\; T\right] $$


Note that this is the *true* error. On the other hand, we have the more mundane training error (or within-sample prediction error). We can also characterize it as the loss function over the training set (i.e., the data points we use to train the model):

$$ \overline{err} = {1 \over N} \sum_{i=1}^N L(y_i, \hat{f} (x_i) )$$

$ \overline{err}$ will generally underestimate the true error. This makes sense: if you use tons of parameters, you're bound to have great *within-sample* prediction (i.e. low training error), but your out-of-sample error will be much larger.

It would be great to get an expression for the difference between $\overline{err} $ and $Err_{T}$, but it seems it's not that easy. Instead of $Err_{T}$, our stats friends turn to something related[^1] 

[^1]: This is called the *in-sample* error, not to be confused with the *within-sample*, which is the way I've called the training error.

$$Err_{in} = {1 \over N} \sum_{i=1}^N E_{Y^0}\; \left[ L(Y_i^0, \hat{f} (x_i) \; |\; T) \right]$$

this is very similar to the out of sample error $Err_{T}$, but not quite. $Err_{T}$ measures the error over new $Ys$ *and* new $Xs$. The $Err_{in}$, instead, measures the error when we observe new response $Y$ values at the *original* training points $x_i, i=1,2,...N$.

With this measure in hand we can define *optimism* as the difference

$$op = Err_{in} - \overline{err}$$

We'll use a related concept: the *average optimism*, taken over training sets:

$$ \omega = E_y (op) = E_y ( Err_{in} - \overline{err}) $$

The size of $\omega$ gives us an idea of how optimistic our within-sample fit is: Will the within-sample value be a good approximation for our out-of-sample fit?

This optimism is connected with the complexity of the model, the degrees of freedom and the number of parameters we use. Students of linear regression know by heart that, with $k$ regressors, the degrees of freedom left from a regression is $n-k$. Once again, The intuition is that the more complicated our model is, the less we should trust the in sample fit as a measure of out of sample fit.

I will show that $\omega$ can be understood as a general concept that can be linked with the $n-k$ expression for Least Squares, but can also be used for other types of estimation techniques, including non-linear or those lacking closed form solutions.

**I'll show this in two steps**

You can find the derivation details below, but here is the summary

1. For some common loss functions (squared error in particular) the *average optimism* of an estimator can be shown to be:

    $$\omega = \frac{2}{N} \sum_{i=1}^N  Cov(\hat{y}_i, y_i)$$

    We can modify this expression a bit to get a general definition of degrees of freedom:

    $$df(\hat{y}) = {\sum_{i=1}^N Cov(\hat{y}_i, y_i) \over \sigma_{\varepsilon}^2} $$

    The $\sigma_{\varepsilon}^2$ is just a normalization, but the real juice is in $\sum_{i=1}^N Cov(\hat{y}_i, y_i)$.
    
    There's nothing wrong in using $y_i$ to predict $\hat{y}_i$. In fact, not using it would mean we are throwing away valuable information. However the more we *depend* on the information contained in $y_i$ to come up with our prediction, the more overly *optimistic* our estimator will be. In the limit, if $\hat{y}_i$ is just $y_i$, you'll have perfect in sample prediction ($R^2 = 1$), but we're pretty sure the out-of-sample prediction is gonna be bad. In this case (it's easy to check by yourself), $df(\hat{y}) = n$.

2. We can show that in the case of linear regression the general definition of degrees of freedom turns out to be our familiar $k$!


    $$ df(\hat{y}) = k $$

    This comes from

    $$ \omega = \frac{2}{N} \sum_{i=1}^N  Cov(\hat{y}_i, y_i) = 2* {k \over N} \sigma_{\varepsilon}^2 $$







-----



### a

This step has somewhat painful algebra, but it's pretty straightforward if we are careful with the notation.

Remember that, by definition, the average optimism is:

$$ \omega = E_y (Err_{in} - \overline{err}) $$

$$ = E_y \left( {1 \over N} \sum_{i=1}^N E_{Y^0} \left[ L(Y_i^0, \hat{f} (x_i) \; |\; T) \right]  
- {1 \over N} \sum_{i=1}^N L(y_i, \hat{f} (x_i) )
\right)$$

Now use a quadratic loss function and expand the squared terms:

$$ = E_y \left( {1 \over N} \sum_{i=1}^N E_{Y^0} \left[ (Y_i^0 - \hat{y}_i)^2 \right]  
- {1 \over N} \sum_{i=1}^N (y_i - \hat{y}_i)^2 )
\right)$$

$$ =  {1 \over N} \sum_{i=1}^N\left(     
E_y E_{Y^0}[(Y_i^0)^2] + E_y E_{Y^0} [(\hat{y}_i^2] -2 E_y E_{Y^0} [Y_i^0 \hat{y}_i] - E_y[y_i^2] - E_y[\hat{y}_i^2] + 2E[y_i \hat{y}_i]
\right)$$

use $E_y E_{Y^0}[(Y_i^0)^2] =  E_y[y_i^2]$ to replace:

$$ =  {1 \over N}\sum_{i=1}^N \left(  E_y[y_i^2]
+ E_y[y_i^2] -2 E_y [y_i] E_y[ \hat{y}_i]
- E_y[y_i^2] - E_y[\hat{y}_i^2] + 2E[y_i \hat{y}_i] 
 \right)$$

$$ =  {2 \over N} \sum_{i=1}^N \left( E[y_i \hat{y}_i]  - E_y [y_i] E_y[ \hat{y}_i]  \right)$$

To finish, note that $Cov(x, w) = E[xw] - E[x w]$, so that:

$$ =  {2 \over N}  \sum_{i=1}^N  Cov(y_i, \hat{y}_i) $$





### b

Remember the relevant part of the degrees of freedom expression comes from $Cov(\hat{y_i}, y_i)$.

We can use the matrix notation for $Cov$ to write

$$  \sum_{i=1}^N  Cov(\hat{y}_i, y_i) =  tr\left( Cov(\hat{y}, y) \right) $$

where $tr$ is the trace of a matrix.

Remember everything is conditioned in $X$ (we are using the *old* training points), but $y$ is still random (we are waiting for new $y$ values). 

In linear regression the predicted $\hat{y}$ are written $\hat{y} = X\hat{\beta} = X(X'X)^{-1}X'y$.

Then:

$$ tr\left( Cov(\hat{y}, y) \right) = tr\left( Cov(X(X'X)^{-1}X'y, y) \right)  $$

$$ = tr\left( X(X'X)^{-1}X'Cov(y, y) \right) = tr\left( X(X'X)^{-1}X'\right) \sigma_{\varepsilon}^2   $$

$$ = tr\left( (X'X)^{-1}X'X\right) \sigma_{\varepsilon}^2 = I_k \sigma_{\varepsilon}^2 = k \sigma_{\varepsilon}^2$$ 

where $k$ is the number of regressors in $X$ or, in linear algebra terms, the number of (independent) columns in $(X'X)$

The (generalized) degrees of freedom formula is normalized by $\sigma_{\varepsilon}^2$, so we end up getting:

$$ df(\hat{y}) = k $$

If you care about the average optimism, in this case it will be:

$$\omega = \frac{2}{N} \sum_{i=1}^N  Cov(\hat{y}_i, y_i) = 2* {k \over N} \sigma_{\varepsilon}^2 $$


Check [this nice handout by Ryan Tishbirani](http://www.stat.cmu.edu/~ryantibs/advmethods/notes/df.pdf) for more details and explanations.
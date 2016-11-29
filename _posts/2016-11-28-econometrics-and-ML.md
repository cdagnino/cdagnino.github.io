---
layout: post
title: Econometrics and Machine Learning - objectives and comparative advantages
date:   2016-11-28 10:11:00
categories: economics machine learning
image: 
---


I was in a seminar a few years ago and the presenter, a fellow PhD student, was arguing in favor of his/her complex model by telling us the Akaike criterion for it
was lower (hence better) than for the simple model.

Someone asked if the presenter had tried with cross-validation instead. The gears in my head started turning: "can these two things ever give different results? I wonder if...". A little googling told me that the asymptotic equivalence (of course, under certain specific conditions) had been established way back in 1977 (Stone: ["An Asymptotic Equivalence of Choice of Model by Cross-validation and Akaike's Criterion"](http://www.stat.washington.edu/courses/stat527/s13/readings/Stone1977.pdf))

This made me wonder. Economists compare models all the time, and yet it seems we (=some of us at least!) are ignorant of some very relevant yet quite old results. Why haven't economists incorporated this knowledge into their toolkit?

The boring answer is because disciplines take time to cross pollinate, and also, you know, this stuff is rather hard, so we need time to digest it.

The more interesting answer is that this is partly driven by how different the objectives of economics and machine learning (or statistical learning) can be.

<i>A warning to the careful reader: while I use the term "economics", the precise comparison should be between the practice of econometrics[^metrix] and the practice of machine learning, since both are ways of analyzing and getting conclusions from data. </i>

[^metrix]: Or, more broadly, the use of empirical analysis in the social sciences

<br>


The stats and machine learning (ML) community have a cool and very developed framework for talking about out-of-sample fit and how it relates to the in-sample fit (see my post [Linear regression and degrees of freedom]({% post_url 2016-11-29-linear-regression-and-df %}) for the formal details). This should come as no surprise, as the name of the game for many of the ML practitioners is providing accurate prediction. 

Shouldn't out-of-sample prediction be equally important to economists? Well, maybe. I don't pretend to be able to define the objective of economics, but it certainly has to do with achieving comprehension or understanding about human beings, by considering them in a particular dimension[^1].

[^1]: What dimension? I would say: humans as participants of a market that try to deal with the scarcity of goods

Getting great out-of-sample prediction may mean you are getting something right, that you somehow picked the right structure of reality. This, however, is just a sign, not an objective. In fact, the empirical estimation of an economic model may have poor out-of-sample prediction, but it may be deemed by economists as a success, because they think it illuminates something or makes us understand something we didn't quite get before.

For many ML practitioners, the opposite is the case. Capturing the underlying reality is a good idea, because the out-of-sample prediction might go up, but once again, this will be a sign of doing a good job, not the objective itself. If a nice black box with no interpretation gives better prediction, they might happily choose this black box.

This diversity in objectives must surely have consequences on the comparative advantages of econometrics and ML. As we know, having comparative advantages means we can can benefit from trade!

What follows is a very rough (and very personal) characterization of what I see are the comparative advantages of these two ways of approaching data. 

#### Empirical Analysis in Economics (or Social Sciences)

1. *Interpretability*: Since the goal is comprehension/interpretation, I expect the results to be easier to communicate, specially for non-specialist audiences. Black boxes can be applied here and there, but it's difficult to illuminate a social mechanism if most of the model is hidden[^experiments]. 
2. *Causality/Prediction*: Some papers are concerned with the effect of just one particular covariate ($X_1$) on the outcome. The predicted outcome or the effect of other covariates can be irrelevant if this effect is identified in a satisfactory manner. While not always, this tends to mean the researchers are after the casual effect of $X_1$ on $Y$.
3. *Time horizon*: Making social science means there's an expectation of some universality. The mechanism identified should work in different times and different places. In fact, the particular data used in the analysis may just be an example and is not particularly interesting[^2]. Economists expect the results to be "true" or useful in different settings[^truth], even if they don't quite fit the current situation. Some might like to describe this as an internal vs external validity tradeoff.

[^experiments]: However, the use of Randomized Controller Trials seems to be an important counterexample in economics. 

[^2]: A surprisingly large percentage of papers on Industrial Organization and Marketing use breakfast cereal data. I'm assuming this is not because of a particular obsession with breakfast, but rather because this data makes it easy to illustrate interesting and more general concepts about firm behavior and market outcomes.

[^truth]: "truth" or usefulness in economics are defined, in part, in reference to economic theory and the current state of the literature. 

#### Machine Learning

1. *Interpretability*: Since the goal is to predict, it may be acceptable for the estimation procedure to be a black box. I'm aware that many ML practitioners have developed tools to make the results easier to communicate, but using a black box is, I think, less frowned upon.
2. *Causality/Prediction*: ML is mostly concerned about predicting the outcome out-of-sample. The exact (causal or otherwise) effects of the covariates on the outcome may not be relevant at all. 
3. *Time horizon*: Less expectation of universality. An application of ML might be more relevant to the data at hand, but the results or insights obtained may be more difficult to translate to different places or time periods.

#### Benefits from trade

[<it> Kleinberg et alii </it> ](https://www.cs.cornell.edu/home/kleinber/aer15-prediction.pdf) argue that the benefits of a policy generally depend on both a prediction and/or a causal problem.
 
When your problem is purely predictive, it makes sense to turn to ML. Causality, on the other hand, is more difficult to investigate without domain knowledge and a theoretical framework. People might dislike using assumptions from economic theory or its knowledge base, but it's hard to claim causality without making some assumptions (strong conclusions require strong assumptions).

Second, ML seems to have an advantage if you're interested in a limited time-horizon and a particular industry. If, instead, you are interested in the long run or want to understand how something might apply in other industries, I think economics might hold an edge.

Of course, the point of "trading" isn't just about dividing up labor, but about combining both econometrics and ML within the same data analysis. [Athey and Imbens' paper](https://arxiv.org/abs/1504.01132) on causality with ML or [Bajari et alii](http://www.nber.org/papers/w20955) on demand estimation are two recent examples from economics.





------
*Thanks to Sebastián Gallegos for helpful suggestions.
Please post your comments below!*







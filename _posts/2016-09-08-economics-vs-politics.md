---
layout: post
title: Politics ain't my business
date:   2016-09-08 18:14:50
categories: economics, politics, policy
image: /assets/article_images/acemoglu_robinson.png
---

>The policy advice I gave made perfect economic sense. It just didn't work out because of the political situation, and I'm not responsible for that

Let's call this the "politics isn't my business" excuse. [*Economics versus Politics: Pitfalls of Policy Advice*](http://economics.mit.edu/files/10403) is Acemoglu's and Robinson's version of a slap on the face for economists that use this excuse.

Everyone agrees that politics do matter; what is interesting about this paper is that is makes a strong case that economists not only should, but actually *can* weave political considerations into their policy advice. They provide a nice (if very basic) framework and tons of examples that should provide a good starting point.

The authors use the example of Unions to explain their point:

>Historically,  unions  have played  a key  role  in  the  creation  of  democracy  in  many  parts  of  the  world,  particularly  in western Europe; they have founded, funded, and supported political parties, such as  the  Labour  Party  in  Britain  or  the  Social  Democratic  parties  of  Scandinavia, which  have  had  large effects  on  public  policy  and  on  the  extent  of  taxation  and income re distribution, often balancing the political power of established business interests  and  political  elites.  Because  the  higher  wages  that  unions  generate  for their members are one of the main reasons why people join unions, reducing their 
market power is likely to foster de-unionization. But this may, by further strengthening groups and interests that were already dominant in society, also change the political equilibrium in a direction involving greater effi ciency losses. This case illustrates a more general conclusion, which is the heart of our argument: even when it 
is possible, removing a market failure need not improve the allocation of resources because of its effect on future political equilibria. To understand whether it is likely to do so, one must look at the political consequences of a policy —it is not sufficient to just focus on the economic costs and benefits.

You can check the paper for more examples and nuances; here I wanna condense the simple model they develop in their footnotes. For those familiar with these authors, you'll recognize these ideas resonate with their broader agenda about the dynamic interplay between economic and political systems. This is a complicated chicken and egg problem, but they manage to throw some light into the situation (see, for example, their "Why Nations Fail" book or ["The colonial origins of comparative development"](http://wikisum.com/w/Acemoglu,_Johnson,_and_Robinson:_The_colonial_origins_of_comparative_development)).

Let's get started! Let $$x_t$$ denote a policy at time $$t$$. 

$$W_t(x_t)$$ is our social welfare function. Let's say we have period 1 and period 2. In the world of economics without politics, the policies $$x_t$$ are chosen independently every period, so that

$$ \large{\max \; W_1(x_1) + W_2(x_2)} $$

gives F.O.C. $$W_1'(x_1^{SW}) = 0$$ where $$x_1^{SW}$$ is the policy that maximizes social welfare.

Now let's introduce politics. The second period policy will be determined by the distribution of political power in that period. Let's call that $$p_2$$, so that the policy $$x_2 = \xi(p_2)$$ is a function of that distribution. In turn, $$p_2 = \pi(x_1)$$, the distribution of political power, is a result of the policy chosen on period one. In other words, the model allows the policies on period 1 to alter the balance of power in period 2.

Once we consider politics, the optimal $$x_1$$ can't be chosen by setting $$W_1'(x_1^*) = 0$$, because we need to take into account the shifting of political power. The right FOC for period's one policy is:

$$ {\Large W_1'(x_1) + W_2'\left(\pi(x_1)\right) {d\xi(\pi(x_1))\over p_2} {d \pi(x_1) \over x_1} = 0 }$$

This looks nice, but how do we use it to think about the unionization example? Acemoglu and Robinson say:

>(...)policies  that  economically  strengthen  already  dominant  groups,  or  that  weaken  those  groups  that  are  acting  as  a  counterbalance  to  the  dominant  groups,  are  especially  likely  to  tilt  the  balance  of  political power further and have unintended, counterproductive implications.

$$W_1'(x_1)$$ is the direct, economic effect of the policy. The second part $$W_2'\left(\pi(x_1)\right) {d\xi(\pi(x_1))\over p_2} {d \pi(x_1) \over x_1}$$ is the unintended effect from the change in the balance of political power.

To be concrete, let's order the policies so that a higher $$x$$ means that the policy tends to favor the most politically powerful groups(i.e. $$x^1_t > x^0_t$$ means that the policy $$x^1$$ favors the powerful groups more than $$x^0$$). 

Call $$x^0_1, x^0_2$$ the status quo policies of the first and second period. Now suppose that the status quo of the second period favors the most powerful groups, so that $$x^2 > x^{SW}_2$$. Any policy in the first period that favors the most powerful groups (i.e. $$x_1 > x_1^0$$) will then shift the political equilibrium further in favour of the powerful group, biasing the second period policy even further away from the social optimum.

Thus, policies that are economically optimal ($$W_1'(x_1)= 0$$ ), may still be disastrous due to the tilting of political power and subsequent policy choices.

My takeway is that any economic policy, however optimal in the strict economic sense, needs to be revised and checked if it tilts the political power to groups that are already powerful. 

Of course, what we really want is an empirical model that allows us to measure the tradeoffs (economic benefits vs political tilt costs), but seeing this simple mechanism is a step in the right direction. At the very least, I expect the readers of the paper to use the "politics isn't my business" excuse feeling some amount of shame.

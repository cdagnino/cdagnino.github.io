---
layout: post
title:  "Active learning of an unknown demand"
date:   2018-10-12 14:56:00
categories: learning
image: /assets/article_images/rstudio_crazy.png
comments: true
---

# Active learning of an unknown demand (& bonus Altair plots)

You're about to launch your product next month, but you're not sure which prices to use.
You think the demand has a log log form:

$$ \log q = \alpha + \beta \log p + \varepsilon $$

but you're unsure of the values of $$\alpha$$ and $$\beta$$

What should your pricing strategy be in this case? How should you go about learning about demand, but without sacrificing too much profit in the meantime?

We can distinguish between two broad types of learning: active and passive.

Passive means you're choosing the optimal price according to your beliefs each period, without thinking about how your beliefs will evolve in the future. You'll still update your beliefs if reality gives you a surprise (say, demand was much higher than what your beliefs expected), but you won't take into account how the way you choose prices affect the rate of learning.

Active means you choose the optimal price as a compromise between exploitation and exploration: do you wanna try to maximize the current profits or do you wanna explore to learn faster / make sure you get the right answer? Thus, in active learning you need to take into account not just the current profits, but also how the current price choice changes the evolution of beliefs. In practice, this involves earning a bit less profit now, but getting more in the long run.

In this blog post, I will present a way of modeling this problem that I find particularly insightful and (relatively!) easy to code.

### Taking a stance

One way to model this problem is through dynamic programming and, more particularly, with a Bellman equation. Thus, the problem of choosing a sequence of prices under active learning can be written as:

$$V_{b_t}(I_t) = max_{p_t \in P} \{ \pi(p_t, x_t) + \beta
                 \int V_{b_{t+1}}(I_{t+1}(x_{t+1}, I_t)) b_t(x_{t+1}| p_t, I_t )\; d x_{t+1}\}  $$


+ $$ \pi(a_t, x_t)$$ is the current period profit
+ $$x_{t+1}$$ in this case is the log demand ($$log q$$)
+ $$I_t$$ represents the information set of the firm at $$t$$
+ $$b_t(x_{t+1} \| p_t, I_t )$$ represents the firm's belief about the value that $$x_{t+1}$$ (log demand) will take next period

To fully flesh out this model, I borrow the notation of Aguirregabiria &amp; Jeon (2018): ["Firms' Belief and Learning in Oligopoly Markets"](http://aguirregabiria.net/wpapers/survey_rio.pdf)

The first important specification is what is the form of the belief function $$b()$$. In their survey paper, Aguirregabiria &amp; Jeon consider four types of learning and belief function

1. Rational expectations
2. Bayesian learning
3. Adaptive learning
4. Reinforcement learning

In this blog post I will only talk about Bayesian learning, but you're welcome to check the paper for the other approaches Under the bayesian learning, the firm starts with some priors on how $$x_{t+1}$$ (log demand) evolves and then updates those priors as new information (i.e. prices chosen and observed demand) comes in.

With [Giovanni Ballarin](https://github.com/giob1994) we are writing the [LearningModel package](https://github.com/cdagnino/LearningModels) that estimates such value functions under different settings. Our idea is to make it easy for a researcher to plug different models and get a value function. For example, it should be easy to change the demand model from

$$ \log q = \alpha + \beta \log p + \varepsilon $$

to an AR(1) one

$$ \log q_{t} = \alpha + \beta \log p_{y} + \gamma q_{t-1} \varepsilon $$

without having to rewrite the value function iteration code completely.


### Example

We'll import the package and do a value function iteration to get the correct value function and policy functions.

With those in hand, we can simulate how would firms learn under a random demand scenario. We'll make all start with the same prior, but their experience with different demand realizations will make them behave and learn in a different way.




## Solve for the value and policy function


```python
!git clone https://github.com/cdagnino/LearningModels.git
!mkdir LearningModels/data

```

    Cloning into 'LearningModels'...
    remote: Enumerating objects: 280, done.[K
    remote: Counting objects: 100% (280/280), done.[K
    remote: Compressing objects: 100% (240/240), done.[K
    remote: Total 280 (delta 147), reused 167 (delta 38), pack-reused 0[K
    Receiving objects: 100% (280/280), 876.49 KiB | 1.11 MiB/s, done.
    Resolving deltas: 100% (147/147), done.



```python
#If you get No module named 'src', you might need to add the folder to your system path
!python LearningModels/examples/aguirregabiria_simple.py
```

After 60 iterations we get an error of 0.004. We could let it run longer to get a smaller error, but it should be fine for our plotting purposes.

## Use the policy function to simulate


```python
%matplotlib inline
import sys
sys.path.append("/Users/cd/Documents/github_reps/cdagnino.github.io/notebooks/LearningModels")

import matplotlib.pyplot as plt
import dill
import numpy as np
import pandas as pd
#file_n = "2018-10-1vfi_dict.dill"  
file_n = "2018-10-11vfi_dict.dill"
with open('LearningModels/data/' + file_n, 'rb') as file:
    data_d = dill.load(file)
    
    
import sys
sys.path.append('../')
import src

lambdas = src.generate_simplex_3dims(n_per_dim=data_d['n_of_lambdas_per_dim'])
price_grid = np.linspace(data_d['min_price'], data_d['max_price'])

policy = data_d['policy']
valueF = data_d['valueF']



lambdas_ext = src.generate_simplex_3dims(n_per_dim=15) #15 should watch value f iteration
print(lambdas_ext.shape)

#Interpolate policy (level price). valueF is already a function
policyF = src.interpolate_wguess(lambdas_ext, policy)

def one_run(lambda0=np.array([0.4, 0.4, 0.2]),
                             true_beta=src.betas_transition[2],
                             dmd_σϵ=src.const.σ_ɛ+0.05, time_periods=40):
    current_lambdas = lambda0
    d = {}
    d['level_prices'] = []
    d['log_dmd'] = []
    d['valueF'] = []
    d['lambda1'] = []
    d['lambda2'] = []
    d['lambda3'] = []
    d['t'] = []


    for t in range(time_periods):
        d['t'].append(t)
        d['lambda1'].append(current_lambdas[0])
        d['lambda2'].append(current_lambdas[1])
        d['lambda3'].append(current_lambdas[2])
        d['valueF'].append(valueF(current_lambdas[:2])[0])

        #0. Choose optimal price (last action of t-1)
        level_price = policyF(current_lambdas[:2]) #Check: Is this correctly defined with the first two elements?
        d['level_prices'].append(level_price[0])

        #1. Demand happens
        log_dmd = src.draw_true_log_dmd(level_price, true_beta, dmd_σε)
        d['log_dmd'].append(log_dmd[0])

        #2. lambda updates: log_dmd: Yes, level_price: Yes
        new_lambdas = src.update_lambdas(log_dmd, src.dmd_transition_fs, current_lambdas,
                       action=level_price, old_state=1.2)

        current_lambdas = new_lambdas
            
    return pd.DataFrame(d)

def many_runs(total_runs, **kwargs):
    dfs = []
    for run in range(total_runs):
        df = one_run(**kwargs)
        df['firm_id'] = run
        dfs.append(df)
        
    return pd.concat(dfs, axis=0)

all_firms = many_runs(7, time_periods=50)



```

    (120, 3)


## Plot with your new BFF: Altair




```python
import altair as alt
all_firms['demand'] = np.e**(all_firms['log_dmd'])
selector = alt.selection_single(empty='all', fields=['firm_id'], on='mouseover')

base = alt.Chart(all_firms).properties(
    width=250,
    height=250
).add_selection(selector).transform_filter(
    selector
)

color_timeseries = alt.Color('firm_id:N', legend=None)

x_for_tseries = alt.X('t', scale=alt.Scale(domain=(0, 50)))

#alt.Y('level_prices', scale=alt.Scale(domain=(0, 3.5)))
timeseries1 = base.mark_line(strokeWidth=2).encode(
    x=x_for_tseries,
    y=alt.Y('level_prices'),
    color=color_timeseries
)

timeseries2 = base.mark_line(strokeWidth=2).encode(
    x=x_for_tseries,
    y=alt.Y('demand'),
    color=color_timeseries 
)

timeseries3 = base.mark_line(strokeWidth=2).encode(
    x=x_for_tseries,
    y=alt.Y('lambda3'),
    color=color_timeseries 
)

timeseries4 = base.mark_line(strokeWidth=2).encode(
    x=x_for_tseries,
    y=alt.Y('valueF', scale=alt.Scale(domain=(14, 22))),
    color=color_timeseries 
)


color = alt.condition(selector,
                      alt.Color('firm_id:N', legend=None, ),
                      alt.value('lightgray'))


legend = alt.Chart(all_firms).mark_point(size=400).encode(
    y=alt.Y('firm_id:N', axis=alt.Axis(orient='right')),
    color=color
).add_selection(
    selector
)


((timeseries1 | timeseries2) & (timeseries3 | timeseries4) | legend).properties(
title='Learning for multiple firms. Hover on firm-id')
```

{% raw %}

  <div id="vis"></div>
  <script src="https://cdn.jsdelivr.net/npm/vega@4.2.0"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-lite@3.0.0-rc6"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-embed@3.19.2"></script>

  <script type="text/javascript">
var yourVlSpec = {
  "config": {"view": {"width": 400, "height": 300}},
  "hconcat": [
    {
      "vconcat": [
        {
          "hconcat": [
            {
              "data": {"name": "data-f1dffa4cd2978d0b3f0d9568c0eb62de"},
              "mark": {"type": "line", "strokeWidth": 2},
              "encoding": {
                "color": {
                  "type": "nominal",
                  "field": "firm_id",
                  "legend": null
                },
                "x": {
                  "type": "quantitative",
                  "field": "t",
                  "scale": {"domain": [0, 50]}
                },
                "y": {"type": "quantitative", "field": "level_prices"}
              },
              "height": 250,
              "selection": {
                "selector043": {
                  "type": "single",
                  "empty": "all",
                  "fields": ["firm_id"],
                  "on": "mouseover",
                  "resolve": "global"
                }
              },
              "transform": [{"filter": {"selection": "selector043"}}],
              "width": 250
            },
            {
              "data": {"name": "data-f1dffa4cd2978d0b3f0d9568c0eb62de"},
              "mark": {"type": "line", "strokeWidth": 2},
              "encoding": {
                "color": {
                  "type": "nominal",
                  "field": "firm_id",
                  "legend": null
                },
                "x": {
                  "type": "quantitative",
                  "field": "t",
                  "scale": {"domain": [0, 50]}
                },
                "y": {"type": "quantitative", "field": "demand"}
              },
              "height": 250,
              "selection": {
                "selector043": {
                  "type": "single",
                  "empty": "all",
                  "fields": ["firm_id"],
                  "on": "mouseover",
                  "resolve": "global"
                }
              },
              "transform": [{"filter": {"selection": "selector043"}}],
              "width": 250
            }
          ]
        },
        {
          "hconcat": [
            {
              "data": {"name": "data-f1dffa4cd2978d0b3f0d9568c0eb62de"},
              "mark": {"type": "line", "strokeWidth": 2},
              "encoding": {
                "color": {
                  "type": "nominal",
                  "field": "firm_id",
                  "legend": null
                },
                "x": {
                  "type": "quantitative",
                  "field": "t",
                  "scale": {"domain": [0, 50]}
                },
                "y": {"type": "quantitative", "field": "lambda3"}
              },
              "height": 250,
              "selection": {
                "selector043": {
                  "type": "single",
                  "empty": "all",
                  "fields": ["firm_id"],
                  "on": "mouseover",
                  "resolve": "global"
                }
              },
              "transform": [{"filter": {"selection": "selector043"}}],
              "width": 250
            },
            {
              "data": {"name": "data-f1dffa4cd2978d0b3f0d9568c0eb62de"},
              "mark": {"type": "line", "strokeWidth": 2},
              "encoding": {
                "color": {
                  "type": "nominal",
                  "field": "firm_id",
                  "legend": null
                },
                "x": {
                  "type": "quantitative",
                  "field": "t",
                  "scale": {"domain": [0, 50]}
                },
                "y": {
                  "type": "quantitative",
                  "field": "valueF",
                  "scale": {"domain": [14, 22]}
                }
              },
              "height": 250,
              "selection": {
                "selector043": {
                  "type": "single",
                  "empty": "all",
                  "fields": ["firm_id"],
                  "on": "mouseover",
                  "resolve": "global"
                }
              },
              "transform": [{"filter": {"selection": "selector043"}}],
              "width": 250
            }
          ]
        }
      ]
    },
    {
      "data": {"name": "data-f1dffa4cd2978d0b3f0d9568c0eb62de"},
      "mark": {"type": "point", "size": 400},
      "encoding": {
        "color": {
          "condition": {
            "type": "nominal",
            "field": "firm_id",
            "legend": null,
            "selection": "selector043"
          },
          "value": "lightgray"
        },
        "y": {
          "type": "nominal",
          "axis": {"orient": "right"},
          "field": "firm_id"
        }
      },
      "selection": {
        "selector043": {
          "type": "single",
          "empty": "all",
          "fields": ["firm_id"],
          "on": "mouseover",
          "resolve": "global"
        }
      }
    }
  ],
  "title": "Learning for multiple firms. Hover on firm-id",
  "$schema": "https://vega.github.io/schema/vega-lite/v2.6.0.json",
  "datasets": {
    "data-f1dffa4cd2978d0b3f0d9568c0eb62de": [
      {
        "level_prices": 0.6896551724137931,
        "log_dmd": 1.7355623648273153,
        "valueF": 19.774116879259957,
        "lambda1": 0.4,
        "lambda2": 0.4,
        "lambda3": 0.2,
        "t": 0,
        "firm_id": 0,
        "demand": 5.6721167059896
      },
      {
        "level_prices": 2.396551724137931,
        "log_dmd": -0.23419323557514354,
        "valueF": 17.910387333041133,
        "lambda1": 0.18777424809465457,
        "lambda2": 0.5672072506429323,
        "lambda3": 0.24501850126241312,
        "t": 1,
        "firm_id": 0,
        "demand": 0.7912089114137112
      },
      {
        "level_prices": 1.6379556564975464,
        "log_dmd": 1.3649892017609444,
        "valueF": 16.7531337439883,
        "lambda1": 0.000009273226348700195,
        "lambda2": 0.6859840191990397,
        "lambda3": 0.3140067075746116,
        "t": 2,
        "firm_id": 0,
        "demand": 3.91568076930742
      },
      {
        "level_prices": 2.205082857012844,
        "log_dmd": 0.39253547437419345,
        "valueF": 17.991989646132417,
        "lambda1": 1.8678239701277824e-9,
        "lambda2": 0.4292545102655842,
        "lambda3": 0.5707454878665917,
        "t": 3,
        "firm_id": 0,
        "demand": 1.480730392333369
      },
      {
        "level_prices": 2.5862068965517064,
        "log_dmd": -0.05792196594101438,
        "valueF": 19.04192743740887,
        "lambda1": 6.646251641951294e-15,
        "lambda2": 0.24276730312240472,
        "lambda3": 0.7572326968775887,
        "t": 4,
        "firm_id": 0,
        "demand": 0.9437235871456239
      },
      {
        "level_prices": 2.76054469012277,
        "log_dmd": -0.22650398394322663,
        "valueF": 19.608589572856353,
        "lambda1": 2.2119462136484173e-21,
        "lambda2": 0.14862602579791773,
        "lambda3": 0.8513739742020823,
        "t": 5,
        "firm_id": 0,
        "demand": 0.7973161658443213
      },
      {
        "level_prices": 2.775862068965517,
        "log_dmd": -0.829621720534412,
        "valueF": 19.941048039772745,
        "lambda1": 2.448512985883755e-28,
        "lambda2": 0.09580432508107604,
        "lambda3": 0.9041956749189239,
        "t": 6,
        "firm_id": 0,
        "demand": 0.43621426601478314
      },
      {
        "level_prices": 2.6832059930146537,
        "log_dmd": 0.415904192678775,
        "valueF": 19.431382875627268,
        "lambda1": 1.783414884538421e-32,
        "lambda2": 0.17775358704642916,
        "lambda3": 0.8222464129535709,
        "t": 7,
        "firm_id": 0,
        "demand": 1.515740642863158
      },
      {
        "level_prices": 2.861902839507742,
        "log_dmd": -0.0887257995439559,
        "valueF": 20.311223586727493,
        "lambda1": 6.351488296431458e-42,
        "lambda2": 0.039023605899681506,
        "lambda3": 0.9609763941003185,
        "t": 8,
        "firm_id": 0,
        "demand": 0.915096459042819
      },
      {
        "level_prices": 2.9230969360091144,
        "log_dmd": -0.4152038269233642,
        "valueF": 20.464937941788072,
        "lambda1": 2.506363921883092e-50,
        "lambda2": 0.015976478645917825,
        "lambda3": 0.9840235213540822,
        "t": 9,
        "firm_id": 0,
        "demand": 0.6602056992933168
      },
      {
        "level_prices": 2.933980100072996,
        "log_dmd": -0.8376457295039706,
        "valueF": 20.492275522383945,
        "lambda1": 1.7131871447479585e-57,
        "lambda2": 0.011877624647832607,
        "lambda3": 0.9881223753521675,
        "t": 10,
        "firm_id": 0,
        "demand": 0.43272808411376973
      },
      {
        "level_prices": 2.9088231163247458,
        "log_dmd": -0.12722993302095797,
        "valueF": 20.429083323246815,
        "lambda1": 1.5807434205729222e-62,
        "lambda2": 0.02135233281275803,
        "lambda3": 0.9786476671872419,
        "t": 11,
        "firm_id": 0,
        "demand": 0.8805311861232127
      },
      {
        "level_prices": 2.9421494570127527,
        "log_dmd": -0.7748808954474302,
        "valueF": 20.51279625077196,
        "lambda1": 4.3704691050085626e-71,
        "lambda2": 0.008800853852339714,
        "lambda3": 0.9911991461476602,
        "t": 12,
        "firm_id": 0,
        "demand": 0.46075865618454626
      },
      {
        "level_prices": 2.928925062630775,
        "log_dmd": -0.45772674569282645,
        "valueF": 20.479577699151594,
        "lambda1": 1.687262860647983e-76,
        "lambda2": 0.013781469918279414,
        "lambda3": 0.9862185300817206,
        "t": 13,
        "firm_id": 0,
        "demand": 0.632720346147381
      },
      {
        "level_prices": 2.9359542021179688,
        "log_dmd": -1.06852420113847,
        "valueF": 20.497234298475362,
        "lambda1": 1.750638409288127e-83,
        "lambda2": 0.011134131669855789,
        "lambda3": 0.9888658683301442,
        "t": 14,
        "firm_id": 0,
        "demand": 0.3435151027152658
      },
      {
        "level_prices": 2.8793350759577936,
        "log_dmd": -0.24093355614061465,
        "valueF": 20.355011879003744,
        "lambda1": 2.5129781124190512e-87,
        "lambda2": 0.032458218145766014,
        "lambda3": 0.967541781854234,
        "t": 15,
        "firm_id": 0,
        "demand": 0.7858938424735683
      },
      {
        "level_prices": 2.918391409279333,
        "log_dmd": -0.2847195559644379,
        "valueF": 20.453118059677834,
        "lambda1": 4.423558356594099e-95,
        "lambda2": 0.017748690011679794,
        "lambda3": 0.9822513099883201,
        "t": 16,
        "firm_id": 0,
        "demand": 0.7522251818069597
      },
      {
        "level_prices": 2.9387831018328283,
        "log_dmd": -0.052607408271008294,
        "valueF": 20.504340253462228,
        "lambda1": 6.837925087335342e-103,
        "lambda2": 0.010068701907116562,
        "lambda3": 0.9899312980928834,
        "t": 17,
        "firm_id": 0,
        "demand": 0.9487524117267884
      },
      {
        "level_prices": 2.9565634282765294,
        "log_dmd": 0.26996960768546263,
        "valueF": 20.54900291813608,
        "lambda1": 4.579509366550302e-112,
        "lambda2": 0.003372215324423847,
        "lambda3": 0.9966277846755762,
        "t": 18,
        "firm_id": 0,
        "demand": 1.309924638486624
      },
      {
        "level_prices": 2.9640697638505222,
        "log_dmd": -0.7915185952298119,
        "valueF": 20.567858193360195,
        "lambda1": 4.4676677735331145e-123,
        "lambda2": 0.0005451538744785568,
        "lambda3": 0.9994548461255215,
        "t": 19,
        "firm_id": 0,
        "demand": 0.45315611178924836
      },
      {
        "level_prices": 2.963203431822034,
        "log_dmd": -0.2095121375805238,
        "valueF": 20.56568204113897,
        "lambda1": 1.539905056394948e-128,
        "lambda2": 0.0008714347683244998,
        "lambda3": 0.9991285652316755,
        "t": 20,
        "firm_id": 0,
        "demand": 0.8109797960407772
      },
      {
        "level_prices": 2.964471073679051,
        "log_dmd": -0.4221437586331197,
        "valueF": 20.568866249447275,
        "lambda1": 4.524573082378048e-137,
        "lambda2": 0.00039401121178599005,
        "lambda3": 0.999605988788214,
        "t": 21,
        "firm_id": 0,
        "demand": 0.6556397787398022
      },
      {
        "level_prices": 2.9647678706777265,
        "log_dmd": -0.6205176981286997,
        "valueF": 20.56961177821302,
        "lambda1": 1.7305936148677064e-144,
        "lambda2": 0.0002822305239730848,
        "lambda3": 0.999717769476027,
        "t": 22,
        "firm_id": 0,
        "demand": 0.5376660168411572
      },
      {
        "level_prices": 2.9646915039404598,
        "log_dmd": -0.5309068710766895,
        "valueF": 20.569419951477467,
        "lambda1": 7.368100206939368e-151,
        "lambda2": 0.0003109920224241196,
        "lambda3": 0.9996890079775759,
        "t": 23,
        "firm_id": 0,
        "demand": 0.5880714228214309
      },
      {
        "level_prices": 2.9647682831303115,
        "log_dmd": -0.12644257459626457,
        "valueF": 20.56961281425876,
        "lambda1": 1.0552319149273267e-157,
        "lambda2": 0.0002820751846877609,
        "lambda3": 0.9997179248153122,
        "t": 24,
        "firm_id": 0,
        "demand": 0.8812247527777559
      },
      {
        "level_prices": 2.9652352123641297,
        "log_dmd": -0.9898923935041735,
        "valueF": 20.57078570069943,
        "lambda1": 1.098416176049901e-166,
        "lambda2": 0.0001062187200029958,
        "lambda3": 0.9998937812799971,
        "t": 25,
        "firm_id": 0,
        "demand": 0.3716166772390472
      },
      {
        "level_prices": 2.9648239624199704,
        "log_dmd": -0.36749413003313636,
        "valueF": 20.569752675889042,
        "lambda1": 4.165709597256096e-171,
        "lambda2": 0.00026110506260852283,
        "lambda3": 0.9997388949373915,
        "t": 26,
        "firm_id": 0,
        "demand": 0.692467391557356
      },
      {
        "level_prices": 2.965076454936583,
        "log_dmd": 0.2631366550930474,
        "valueF": 20.570386915571753,
        "lambda1": 8.146134379844928e-179,
        "lambda2": 0.00016601047842956806,
        "lambda3": 0.9998339895215704,
        "t": 27,
        "firm_id": 0,
        "demand": 1.3010044956948132
      },
      {
        "level_prices": 2.9654461077654886,
        "log_dmd": -1.2849147908801157,
        "valueF": 20.57131545197379,
        "lambda1": 7.352560174631548e-190,
        "lambda2": 0.00002679058182888458,
        "lambda3": 0.9999732094181711,
        "t": 28,
        "firm_id": 0,
        "demand": 0.2766741577890725
      },
      {
        "level_prices": 2.9651851969762673,
        "log_dmd": -0.42961239461901474,
        "valueF": 20.570660066307745,
        "lambda1": 1.009304441058533e-192,
        "lambda2": 0.00012505568426293792,
        "lambda3": 0.999874944315737,
        "t": 29,
        "firm_id": 0,
        "demand": 0.6507612844206957
      },
      {
        "level_prices": 2.965275724705774,
        "log_dmd": -0.16927864296817788,
        "valueF": 20.57088746424892,
        "lambda1": 4.179626027703334e-200,
        "lambda2": 0.00009096082509784058,
        "lambda3": 0.9999090391749021,
        "t": 30,
        "firm_id": 0,
        "demand": 0.8442736197001368
      },
      {
        "level_prices": 2.965417500813198,
        "log_dmd": -0.3730367750271629,
        "valueF": 20.571243593747703,
        "lambda1": 7.264670031355033e-209,
        "lambda2": 0.00003756462879536175,
        "lambda3": 0.9999624353712045,
        "t": 31,
        "firm_id": 0,
        "demand": 0.6886399076269825
      },
      {
        "level_prices": 2.9654531086854456,
        "log_dmd": 0.3400601538465031,
        "valueF": 20.57133303768806,
        "lambda1": 1.5051414553059724e-216,
        "lambda2": 0.00002415387171518038,
        "lambda3": 0.9999758461282847,
        "t": 32,
        "firm_id": 0,
        "demand": 1.4050321061072435
      },
      {
        "level_prices": 2.9655084919758945,
        "log_dmd": -0.8148670236259307,
        "valueF": 20.57147215579355,
        "lambda1": 5.287484970248943e-228,
        "lambda2": 0.000003295229857761444,
        "lambda3": 0.9999967047701424,
        "t": 33,
        "firm_id": 0,
        "demand": 0.4426981918478304
      },
      {
        "level_prices": 2.9655025417201815,
        "log_dmd": -0.7864571913273732,
        "valueF": 20.571457209258316,
        "lambda1": 2.372177807097097e-233,
        "lambda2": 0.000005536235256259823,
        "lambda3": 0.9999944637647438,
        "t": 34,
        "firm_id": 0,
        "demand": 0.45545553214060164
      },
      {
        "level_prices": 2.9654940239821412,
        "log_dmd": -0.8286751237158223,
        "valueF": 20.571435813426298,
        "lambda1": 7.531220295457756e-239,
        "lambda2": 0.000008744214518139151,
        "lambda3": 0.9999912557854818,
        "t": 35,
        "firm_id": 0,
        "demand": 0.4366273805468196
      },
      {
        "level_prices": 2.965477045295385,
        "log_dmd": -0.13669268386738412,
        "valueF": 20.5713931644122,
        "lambda1": 3.9981847946436916e-244,
        "lambda2": 0.000015138784854764743,
        "lambda3": 0.9999848612151453,
        "t": 36,
        "firm_id": 0,
        "demand": 0.8722382378298105
      },
      {
        "level_prices": 2.9655017815368514,
        "log_dmd": 0.19935565254655863,
        "valueF": 20.57145529974258,
        "lambda1": 4.657413636512994e-253,
        "lambda2": 0.000005822538068829249,
        "lambda3": 0.9999941774619312,
        "t": 37,
        "firm_id": 0,
        "demand": 1.220616003902095
      },
      {
        "level_prices": 2.9655143777673256,
        "log_dmd": 0.3499964508348633,
        "valueF": 20.5714869404,
        "lambda1": 9.06538100406436e-264,
        "lambda2": 0.0000010785032148512382,
        "lambda3": 0.9999989214967852,
        "t": 38,
        "firm_id": 0,
        "demand": 1.4190625120971248
      },
      {
        "level_prices": 2.965516859102551,
        "log_dmd": -0.477117182880387,
        "valueF": 20.571493173302613,
        "lambda1": 2.8184421267678403e-275,
        "lambda2": 1.4397436380618286e-7,
        "lambda3": 0.9999998560256361,
        "t": 39,
        "firm_id": 0,
        "demand": 0.6205698048752757
      },
      {
        "level_prices": 2.965516933201931,
        "log_dmd": -0.6190573355345725,
        "valueF": 20.571493359433937,
        "lambda1": 2.070117462438818e-282,
        "lambda2": 1.1606680511778754e-7,
        "lambda3": 0.9999998839331948,
        "t": 40,
        "firm_id": 0,
        "demand": 0.5384517777885864
      },
      {
        "level_prices": 2.965516903124817,
        "log_dmd": -0.15636201948843906,
        "valueF": 20.571493283882795,
        "lambda1": 8.560366850224929e-289,
        "lambda2": 1.2739454930467282e-7,
        "lambda3": 0.9999998726054508,
        "t": 41,
        "firm_id": 0,
        "demand": 0.8552495173278115
      },
      {
        "level_prices": 2.9655171056076424,
        "log_dmd": 0.06378189993087513,
        "valueF": 20.571493792502395,
        "lambda1": 1.2661267113665332e-297,
        "lambda2": 5.113478376111272e-8,
        "lambda3": 0.9999999488652161,
        "t": 42,
        "firm_id": 0,
        "demand": 1.0658599092895484
      },
      {
        "level_prices": 2.965517207610597,
        "log_dmd": -1.0841472673917032,
        "valueF": 20.571494048725118,
        "lambda1": 1.283642271835646e-307,
        "lambda2": 1.2718086850410057e-8,
        "lambda3": 0.9999999872819131,
        "t": 43,
        "firm_id": 0,
        "demand": 0.3381900486540069
      },
      {
        "level_prices": 2.9655171394977633,
        "log_dmd": -0.40403158662230676,
        "valueF": 20.57149387763149,
        "lambda1": 1.527967893386e-311,
        "lambda2": 3.837097231336273e-8,
        "lambda3": 0.9999999616290276,
        "t": 44,
        "firm_id": 0,
        "demand": 0.6676230329792907
      },
      {
        "level_prices": 2.9655171713127553,
        "log_dmd": 0.1860841452661265,
        "valueF": 20.57149395754804,
        "lambda1": 4.6095e-319,
        "lambda2": 2.6388702539444983e-8,
        "lambda3": 0.9999999736112974,
        "t": 45,
        "firm_id": 0,
        "demand": 1.2045236110732886
      },
      {
        "level_prices": 2.9655172280214104,
        "log_dmd": -0.12550150511896935,
        "valueF": 20.571494099995345,
        "lambda1": 0,
        "lambda2": 5.0308974164847435e-9,
        "lambda3": 0.9999999949691025,
        "t": 46,
        "firm_id": 0,
        "demand": 0.882054436829169
      },
      {
        "level_prices": 2.965517236365533,
        "log_dmd": -0.18883017726722132,
        "valueF": 20.57149412095507,
        "lambda1": 0,
        "lambda2": 1.8883056459454976e-9,
        "lambda3": 0.9999999981116944,
        "t": 47,
        "firm_id": 0,
        "demand": 0.8279270955987159
      },
      {
        "level_prices": 2.9655172392196425,
        "log_dmd": 0.32863464867670567,
        "valueF": 20.571494128124353,
        "lambda1": 0,
        "lambda2": 8.133811532196213e-10,
        "lambda3": 0.9999999991866189,
        "t": 48,
        "firm_id": 0,
        "demand": 1.389070264209488
      },
      {
        "level_prices": 2.9655172410773023,
        "log_dmd": -0.4547011467410318,
        "valueF": 20.571494132790637,
        "lambda1": 0,
        "lambda2": 1.13743125899412e-10,
        "lambda3": 0.9999999998862569,
        "t": 49,
        "firm_id": 0,
        "demand": 0.6346376031262293
      },
      {
        "level_prices": 0.6896551724137931,
        "log_dmd": 1.9497317347108287,
        "valueF": 19.774116879259957,
        "lambda1": 0.4,
        "lambda2": 0.4,
        "lambda3": 0.2,
        "t": 0,
        "firm_id": 1,
        "demand": 7.026802280574574
      },
      {
        "level_prices": 1.019753498212179,
        "log_dmd": 0.7823852989769817,
        "valueF": 18.996562850571596,
        "lambda1": 0.33469544088078484,
        "lambda2": 0.4861863615466765,
        "lambda3": 0.17911819757253866,
        "t": 1,
        "firm_id": 1,
        "demand": 2.1866819399068245
      },
      {
        "level_prices": 1.059647805719321,
        "log_dmd": 1.3290562549209295,
        "valueF": 19.06150590631767,
        "lambda1": 0.34165975993382347,
        "lambda2": 0.4820543695090984,
        "lambda3": 0.17628587055707803,
        "t": 2,
        "firm_id": 1,
        "demand": 3.777476729750614
      },
      {
        "level_prices": 0.7677179794371786,
        "log_dmd": 1.5217699990980793,
        "valueF": 18.51712214636429,
        "lambda1": 0.28244758816208054,
        "lambda2": 0.5186028287395243,
        "lambda3": 0.19894958309839514,
        "t": 3,
        "firm_id": 1,
        "demand": 4.580325195996173
      },
      {
        "level_prices": 2.3206812734570046,
        "log_dmd": -0.3736749885515602,
        "valueF": 17.756319006791546,
        "lambda1": 0.18606991529471376,
        "lambda2": 0.6000031567499593,
        "lambda3": 0.21392692795532695,
        "t": 4,
        "firm_id": 1,
        "demand": 0.6882005485418635
      },
      {
        "level_prices": 1.453582178263356,
        "log_dmd": 0.16473559532338772,
        "valueF": 16.365413594535465,
        "lambda1": 0.00008627985092347447,
        "lambda2": 0.78388836269357,
        "lambda3": 0.21602535745550663,
        "t": 5,
        "firm_id": 1,
        "demand": 1.1790813228771937
      },
      {
        "level_prices": 1.3609374457368653,
        "log_dmd": 0.486225374226398,
        "valueF": 16.24907854163581,
        "lambda1": 0.00004073727735971016,
        "lambda2": 0.8187301868142987,
        "lambda3": 0.18122907590834164,
        "t": 6,
        "firm_id": 1,
        "demand": 1.6261664509209377
      },
      {
        "level_prices": 1.3446342291985414,
        "log_dmd": -0.18379959938508317,
        "valueF": 16.228944631203092,
        "lambda1": 0.000014600151281301814,
        "lambda2": 0.824791947898549,
        "lambda3": 0.17519345195016964,
        "t": 7,
        "firm_id": 1,
        "demand": 0.8321025409798859
      },
      {
        "level_prices": 1.2589340030667062,
        "log_dmd": 0.5595955481806573,
        "valueF": 16.061789103240734,
        "lambda1": 0.00003933371833110389,
        "lambda2": 0.8803463319145106,
        "lambda3": 0.11961433436715836,
        "t": 8,
        "firm_id": 1,
        "demand": 1.7499645807875999
      },
      {
        "level_prices": 1.2588201652555102,
        "log_dmd": 1.150449583916168,
        "valueF": 16.048546442703454,
        "lambda1": 0.000025042391384393836,
        "lambda2": 0.8854292175196028,
        "lambda3": 0.11454574008901283,
        "t": 9,
        "firm_id": 1,
        "demand": 3.1596131016499296
      },
      {
        "level_prices": 1.2586558447480705,
        "log_dmd": 1.0213633660621786,
        "valueF": 16.11235396789349,
        "lambda1": 0.000004413409930898831,
        "lambda2": 0.8608255558107951,
        "lambda3": 0.13917003077927392,
        "t": 10,
        "firm_id": 1,
        "demand": 2.776978222122476
      },
      {
        "level_prices": 1.3039096779178778,
        "log_dmd": 0.1894105315893101,
        "valueF": 16.17830706530659,
        "lambda1": 0.0000010278579013694184,
        "lambda2": 0.8400890490332046,
        "lambda3": 0.159909923108894,
        "t": 11,
        "firm_id": 1,
        "demand": 1.2085369932683157
      },
      {
        "level_prices": 1.2586303499933449,
        "log_dmd": 0.5436733571986873,
        "valueF": 16.093274964268222,
        "lambda1": 0.0000012127697272790193,
        "lambda2": 0.86817047204118,
        "lambda3": 0.13182831518909285,
        "t": 12,
        "firm_id": 1,
        "demand": 1.7223219600490711
      },
      {
        "level_prices": 1.2586270727661386,
        "log_dmd": 0.5675716370020495,
        "valueF": 16.07683410484337,
        "lambda1": 8.013429351416364e-7,
        "lambda2": 0.8745027255040098,
        "lambda3": 0.125496473153055,
        "t": 13,
        "firm_id": 1,
        "demand": 1.7639782666262658
      },
      {
        "level_prices": 1.2586246927185167,
        "log_dmd": 0.8999108039756626,
        "valueF": 16.064088763603383,
        "lambda1": 5.025490778513515e-7,
        "lambda2": 0.8794116628146441,
        "lambda3": 0.12058783463627809,
        "t": 14,
        "firm_id": 1,
        "demand": 2.4593837341219116
      },
      {
        "level_prices": 1.2586219086347483,
        "log_dmd": 1.1555694867205246,
        "valueF": 16.094878041622692,
        "lambda1": 1.530320679816358e-7,
        "lambda2": 0.8675516569696378,
        "lambda3": 0.13244818999829422,
        "t": 15,
        "firm_id": 1,
        "demand": 3.175831496504134
      },
      {
        "level_prices": 1.3054763838834893,
        "log_dmd": 1.275981136741433,
        "valueF": 16.180265270187014,
        "lambda1": 2.6610778364716215e-8,
        "lambda2": 0.839495987200891,
        "lambda3": 0.1605039861883306,
        "t": 16,
        "firm_id": 1,
        "demand": 3.5822143285865424
      },
      {
        "level_prices": 1.447384640209653,
        "log_dmd": -0.0950974055898931,
        "valueF": 16.35703117204707,
        "lambda1": 1.9651284600828318e-9,
        "lambda2": 0.786049946595646,
        "lambda3": 0.2139500514392256,
        "t": 17,
        "firm_id": 1,
        "demand": 0.9092843607749732
      },
      {
        "level_prices": 1.2843948360221797,
        "log_dmd": 1.199023497468012,
        "valueF": 16.154005539905068,
        "lambda1": 2.4380523411072375e-9,
        "lambda2": 0.8474357184226867,
        "lambda3": 0.15256427913926093,
        "t": 18,
        "firm_id": 1,
        "demand": 3.3168764026005597
      },
      {
        "level_prices": 1.3906421927215247,
        "log_dmd": 1.002610469288142,
        "valueF": 16.286350865838386,
        "lambda1": 2.8422897681849603e-10,
        "lambda2": 0.807420473723801,
        "lambda3": 0.1925795259919701,
        "t": 19,
        "firm_id": 1,
        "demand": 2.725387089686932
      },
      {
        "level_prices": 1.534360354576033,
        "log_dmd": 0.41259911707058977,
        "valueF": 16.48239796974959,
        "lambda1": 1.519803215329261e-11,
        "lambda2": 0.7532928535017601,
        "lambda3": 0.24670714648304182,
        "t": 20,
        "firm_id": 1,
        "demand": 1.5107392749383768
      },
      {
        "level_prices": 1.5485583557678595,
        "log_dmd": 0.706760282091206,
        "valueF": 16.50289174275912,
        "lambda1": 1.2485356982158609e-12,
        "lambda2": 0.7479455543236928,
        "lambda3": 0.25205444567505864,
        "t": 21,
        "firm_id": 1,
        "demand": 2.0274123633764103
      },
      {
        "level_prices": 1.6379310344828242,
        "log_dmd": 0.5963510030271637,
        "valueF": 16.74562249836857,
        "lambda1": 2.4803053691682867e-14,
        "lambda2": 0.6877313048133687,
        "lambda3": 0.31226869518660644,
        "t": 22,
        "firm_id": 1,
        "demand": 1.815482010580867
      },
      {
        "level_prices": 1.703472713455638,
        "log_dmd": 1.0770637037801285,
        "valueF": 17.052537988503868,
        "lambda1": 2.3522121253022583e-16,
        "lambda2": 0.618172614412812,
        "lambda3": 0.38182738558718776,
        "t": 23,
        "firm_id": 1,
        "demand": 2.93604578148673
      },
      {
        "level_prices": 2.293998306840885,
        "log_dmd": -0.0577276546247974,
        "valueF": 18.174706552534367,
        "lambda1": 6.065009869456138e-20,
        "lambda2": 0.3957668714495368,
        "lambda3": 0.6042331285504633,
        "t": 24,
        "firm_id": 1,
        "demand": 0.9439069811351739
      },
      {
        "level_prices": 2.4436308046271904,
        "log_dmd": 0.5856940192300453,
        "valueF": 18.48597106905511,
        "lambda1": 2.3704997264686517e-24,
        "lambda2": 0.3394117748806686,
        "lambda3": 0.6605882251193315,
        "t": 25,
        "firm_id": 1,
        "demand": 1.7962371762918101
      },
      {
        "level_prices": 2.775862068965517,
        "log_dmd": -0.5702132237816213,
        "valueF": 19.93882226587155,
        "lambda1": 1.6426914251443706e-32,
        "lambda2": 0.09615651889585981,
        "lambda3": 0.9038434811041401,
        "t": 26,
        "firm_id": 1,
        "demand": 0.5654048680816129
      },
      {
        "level_prices": 2.775862068965517,
        "log_dmd": 0.0811934835676435,
        "valueF": 19.830357611414115,
        "lambda1": 6.646656103216835e-38,
        "lambda2": 0.11331934949693433,
        "lambda3": 0.8866806505030658,
        "t": 27,
        "firm_id": 1,
        "demand": 1.0845807248149812
      },
      {
        "level_prices": 2.8619237483371505,
        "log_dmd": -1.403126614219316,
        "valueF": 20.311276107924982,
        "lambda1": 1.7298287135527545e-46,
        "lambda2": 0.03901573114574845,
        "lambda3": 0.9609842688542515,
        "t": 28,
        "firm_id": 1,
        "demand": 0.24582715444348788
      },
      {
        "level_prices": 2.6114112210857274,
        "log_dmd": 0.11623117105895853,
        "valueF": 19.26687901290956,
        "lambda1": 2.91126998918547e-48,
        "lambda2": 0.20479317647420658,
        "lambda3": 0.7952068235257933,
        "t": 29,
        "firm_id": 1,
        "demand": 1.1232555062870166
      },
      {
        "level_prices": 2.775862068965517,
        "log_dmd": -0.21899331849117099,
        "valueF": 19.987159213059083,
        "lambda1": 9.935027660099509e-56,
        "lambda2": 0.0885079550341798,
        "lambda3": 0.9114920449658203,
        "t": 30,
        "firm_id": 1,
        "demand": 0.8033270855727102
      },
      {
        "level_prices": 2.8224755386067732,
        "log_dmd": -1.2973686775848567,
        "valueF": 20.212185567110097,
        "lambda1": 7.665863883138549e-63,
        "lambda2": 0.05387284909615023,
        "lambda3": 0.9461271509038498,
        "t": 31,
        "firm_id": 1,
        "demand": 0.27324985636361554
      },
      {
        "level_prices": 2.586206896551724,
        "log_dmd": -1.049138703403563,
        "valueF": 19.12945780136795,
        "lambda1": 5.961049509518868e-65,
        "lambda2": 0.22785705823869043,
        "lambda3": 0.7721429417613095,
        "t": 32,
        "firm_id": 1,
        "demand": 0.35023927913818226
      },
      {
        "level_prices": 1.9804827578548712,
        "log_dmd": -0.3623109571879136,
        "valueF": 17.55460943805234,
        "lambda1": 4.235855567834444e-67,
        "lambda2": 0.5138441561325809,
        "lambda3": 0.486155843867419,
        "t": 33,
        "firm_id": 1,
        "demand": 0.6960658874944775
      },
      {
        "level_prices": 1.653052411710919,
        "log_dmd": 0.1664448643390191,
        "valueF": 16.964279818758534,
        "lambda1": 1.3260650521476016e-68,
        "lambda2": 0.6371620787062772,
        "lambda3": 0.3628379212937228,
        "t": 34,
        "firm_id": 1,
        "demand": 1.1810984134333977
      },
      {
        "level_prices": 1.6379310344827585,
        "log_dmd": 0.24455012029813583,
        "valueF": 16.85951804755848,
        "lambda1": 8.626403186692392e-70,
        "lambda2": 0.661137739485595,
        "lambda3": 0.33886226051440493,
        "t": 35,
        "firm_id": 1,
        "demand": 1.2770466665800249
      },
      {
        "level_prices": 1.6379310344827582,
        "log_dmd": 0.1480475112013971,
        "valueF": 16.821913623831065,
        "lambda1": 4.554305684203461e-71,
        "lambda2": 0.6699180272678699,
        "lambda3": 0.33008197273213014,
        "t": 36,
        "firm_id": 1,
        "demand": 1.1595679875224287
      },
      {
        "level_prices": 1.6379310344827585,
        "log_dmd": 0.24724189105484798,
        "valueF": 16.697401201250525,
        "lambda1": 3.8369956220981387e-72,
        "lambda2": 0.6989905350837077,
        "lambda3": 0.3010094649162923,
        "t": 37,
        "firm_id": 1,
        "demand": 1.280488814109735
      },
      {
        "level_prices": 1.6379310344827587,
        "log_dmd": 0.057743456543302873,
        "valueF": 16.664494205815316,
        "lambda1": 1.9966179037738053e-73,
        "lambda2": 0.7066740164758317,
        "lambda3": 0.2933259835241683,
        "t": 38,
        "firm_id": 1,
        "demand": 1.0594431676315226
      },
      {
        "level_prices": 1.5408796012471564,
        "log_dmd": 1.286419900645213,
        "valueF": 16.491808024040054,
        "lambda1": 2.580937786460037e-74,
        "lambda2": 0.7508375527770449,
        "lambda3": 0.24916244722295514,
        "t": 39,
        "firm_id": 1,
        "demand": 3.619804072105636
      },
      {
        "level_prices": 1.810946952509992,
        "log_dmd": 0.28743517388456014,
        "valueF": 17.24066616912713,
        "lambda1": 4.726551063488544e-77,
        "lambda2": 0.5776953036001328,
        "lambda3": 0.4223046963998672,
        "t": 40,
        "firm_id": 1,
        "demand": 1.3330041758796296
      },
      {
        "level_prices": 1.9264930629450094,
        "log_dmd": 0.5814869092515058,
        "valueF": 17.454036937680435,
        "lambda1": 2.06763199727825e-79,
        "lambda2": 0.534177937332399,
        "lambda3": 0.4658220626676009,
        "t": 41,
        "firm_id": 1,
        "demand": 1.7886960831706362
      },
      {
        "level_prices": 2.380205927728204,
        "log_dmd": 0.127499958355791,
        "valueF": 18.352022403901387,
        "lambda1": 2.2317042681619954e-83,
        "lambda2": 0.3632990661802866,
        "lambda3": 0.6367009338197134,
        "t": 42,
        "firm_id": 1,
        "demand": 1.1359848209339716
      },
      {
        "level_prices": 2.586206896551724,
        "log_dmd": -0.01926816557872152,
        "valueF": 19.137928439561374,
        "lambda1": 4.008980841519108e-89,
        "lambda2": 0.2264141385012109,
        "lambda3": 0.7735858614987892,
        "t": 43,
        "firm_id": 1,
        "demand": 0.9809162789878138
      },
      {
        "level_prices": 2.775862068965517,
        "log_dmd": 0.07606653992487583,
        "valueF": 19.731366031960476,
        "lambda1": 8.855386325493181e-96,
        "lambda2": 0.12898321453752584,
        "lambda3": 0.8710167854624741,
        "t": 44,
        "firm_id": 1,
        "demand": 1.0790343706345518
      },
      {
        "level_prices": 2.8450239569751385,
        "log_dmd": -0.09028944734184424,
        "valueF": 20.268825272331526,
        "lambda1": 2.471284730861949e-104,
        "lambda2": 0.04538058763274004,
        "lambda3": 0.95461941236726,
        "t": 45,
        "firm_id": 1,
        "demand": 0.9136666885997987
      },
      {
        "level_prices": 2.914695500535215,
        "log_dmd": -0.45203471542023566,
        "valueF": 20.443834251901507,
        "lambda1": 1.330696849420479e-112,
        "lambda2": 0.01914065564258106,
        "lambda3": 0.980859344357419,
        "t": 46,
        "firm_id": 1,
        "demand": 0.6363320788061163
      },
      {
        "level_prices": 2.9242566579315157,
        "log_dmd": 0.5849963373893331,
        "valueF": 20.46785106441825,
        "lambda1": 1.616015711701469e-119,
        "lambda2": 0.015539700259558952,
        "lambda3": 0.9844602997404411,
        "t": 47,
        "firm_id": 1,
        "demand": 1.7949844112988087
      },
      {
        "level_prices": 2.961899925124745,
        "log_dmd": -0.40878548093611744,
        "valueF": 20.562407743408524,
        "lambda1": 6.594156448815899e-132,
        "lambda2": 0.001362365862108873,
        "lambda3": 0.9986376341378911,
        "t": 48,
        "firm_id": 1,
        "demand": 0.6644567556759001
      },
      {
        "level_prices": 2.9629919251599794,
        "log_dmd": -0.7178883675350011,
        "valueF": 20.565150754429837,
        "lambda1": 2.2345599535113948e-139,
        "lambda2": 0.0009510931215660408,
        "lambda3": 0.999048906878434,
        "t": 49,
        "firm_id": 1,
        "demand": 0.48778118380245467
      },
      {
        "level_prices": 0.6896551724137931,
        "log_dmd": 2.8012077123709522,
        "valueF": 19.774116879259957,
        "lambda1": 0.4,
        "lambda2": 0.4,
        "lambda3": 0.2,
        "t": 0,
        "firm_id": 2,
        "demand": 16.464519172095265
      },
      {
        "level_prices": 0.6896551724137931,
        "log_dmd": 2.2272559681801383,
        "valueF": 24.826380511353765,
        "lambda1": 0.9136146444611765,
        "lambda2": 0.07224820057525201,
        "lambda3": 0.014137154963571466,
        "t": 1,
        "firm_id": 2,
        "demand": 9.274381931687941
      },
      {
        "level_prices": 0.6896551724137931,
        "log_dmd": 0.7144619269722684,
        "valueF": 25.24910659575464,
        "lambda1": 0.95265046458982,
        "lambda2": 0.04237815598443357,
        "lambda3": 0.0049713794257465254,
        "t": 2,
        "firm_id": 2,
        "demand": 2.0430870564072214
      },
      {
        "level_prices": 2.206896551724138,
        "log_dmd": -0.870003523380358,
        "valueF": 17.267931907109208,
        "lambda1": 0.15718494840161187,
        "lambda2": 0.6928451765440263,
        "lambda3": 0.14996987505436188,
        "t": 3,
        "firm_id": 2,
        "demand": 0.4189500731245799
      },
      {
        "level_prices": 1.338542602937196,
        "log_dmd": 0.5351821310118284,
        "valueF": 15.990196830430367,
        "lambda1": 0.01260057349804168,
        "lambda2": 0.9236721147210354,
        "lambda3": 0.06372731178092296,
        "t": 4,
        "firm_id": 2,
        "demand": 1.7077592498514225
      },
      {
        "level_prices": 1.2839142200898956,
        "log_dmd": 0.7708867738670955,
        "valueF": 15.943358175232852,
        "lambda1": 0.0047630674195258075,
        "lambda2": 0.932460745563532,
        "lambda3": 0.06277618701694221,
        "t": 5,
        "firm_id": 2,
        "demand": 2.161682327494291
      },
      {
        "level_prices": 1.2669400322898359,
        "log_dmd": 0.8351903329290963,
        "valueF": 15.935295752837353,
        "lambda1": 0.0015666294571768959,
        "lambda2": 0.9314090952088041,
        "lambda3": 0.06702427533401888,
        "t": 6,
        "firm_id": 2,
        "demand": 2.30525277204631
      },
      {
        "level_prices": 1.2627171417873495,
        "log_dmd": 0.03208017653653139,
        "valueF": 15.942443983196714,
        "lambda1": 0.0005142732113988588,
        "lambda2": 0.9269140530368869,
        "lambda3": 0.07257167375171426,
        "t": 7,
        "firm_id": 2,
        "demand": 1.0326002923013449
      },
      {
        "level_prices": 1.2639504939123076,
        "log_dmd": 0.030460186263533084,
        "valueF": 15.905933494321495,
        "lambda1": 0.001003664438031968,
        "lambda2": 0.9439060832426249,
        "lambda3": 0.05509025231934312,
        "t": 8,
        "firm_id": 2,
        "demand": 1.0309288441023747
      },
      {
        "level_prices": 1.2689736162408956,
        "log_dmd": 0.7220962644606495,
        "valueF": 15.881277163317547,
        "lambda1": 0.001949577084324473,
        "lambda2": 0.9564909894506501,
        "lambda3": 0.04155943346502532,
        "t": 9,
        "firm_id": 2,
        "demand": 2.0587443630492075
      },
      {
        "level_prices": 1.262919223939897,
        "log_dmd": 0.4311127635605716,
        "valueF": 15.878034776231798,
        "lambda1": 0.0008094642484221691,
        "lambda2": 0.9562823026259878,
        "lambda3": 0.04290823312559021,
        "t": 10,
        "firm_id": 2,
        "demand": 1.5389690798058078
      },
      {
        "level_prices": 1.2621314810348467,
        "log_dmd": -0.10126222325085554,
        "valueF": 15.867970590903202,
        "lambda1": 0.0006611230520165949,
        "lambda2": 0.9606315226728073,
        "lambda3": 0.038707354275176103,
        "t": 11,
        "firm_id": 2,
        "demand": 0.9036960317023934
      },
      {
        "level_prices": 1.2676825391776472,
        "log_dmd": 1.6998385039733868,
        "valueF": 15.848786202788437,
        "lambda1": 0.001706452182803678,
        "lambda2": 0.970879391092217,
        "lambda3": 0.027414156724979374,
        "t": 12,
        "firm_id": 2,
        "demand": 5.473063442352623
      },
      {
        "level_prices": 1.2590637639484148,
        "log_dmd": 1.1624653582595195,
        "valueF": 15.877085396396936,
        "lambda1": 0.00008343606820799769,
        "lambda2": 0.9556431687473431,
        "lambda3": 0.044273395184448804,
        "t": 13,
        "firm_id": 2,
        "demand": 3.1978073066306654
      },
      {
        "level_prices": 1.258697911133301,
        "log_dmd": 0.5882192356513651,
        "valueF": 15.900411190922418,
        "lambda1": 0.000014541706920359581,
        "lambda2": 0.9449547168640652,
        "lambda3": 0.0550307414290144,
        "t": 14,
        "firm_id": 2,
        "demand": 1.8007787957786359
      },
      {
        "level_prices": 1.2586668571560036,
        "log_dmd": 0.7865001834018587,
        "valueF": 15.896324158819573,
        "lambda1": 0.000008693880026652188,
        "lambda2": 0.9468010858346303,
        "lambda3": 0.053190220285343134,
        "t": 15,
        "firm_id": 2,
        "demand": 2.19569842153095
      },
      {
        "level_prices": 1.2586387452544023,
        "log_dmd": 0.264128145565116,
        "valueF": 15.902579754245771,
        "lambda1": 0.000003400080374466069,
        "lambda2": 0.9439540469928093,
        "lambda3": 0.05604255292681635,
        "t": 16,
        "firm_id": 2,
        "demand": 1.302295068946229
      },
      {
        "level_prices": 1.2586422935603576,
        "log_dmd": 0.13049533191683338,
        "valueF": 15.882699874833236,
        "lambda1": 0.000004068267859568187,
        "lambda2": 0.9529779433782729,
        "lambda3": 0.0470179883538674,
        "t": 17,
        "firm_id": 2,
        "demand": 1.1393926211016316
      },
      {
        "level_prices": 1.2586550154470666,
        "log_dmd": -0.07869744384389454,
        "valueF": 15.860915178821328,
        "lambda1": 0.00000646394782421562,
        "lambda2": 0.9628689278206752,
        "lambda3": 0.037124608231500586,
        "t": 18,
        "firm_id": 2,
        "demand": 0.9243195407103931
      },
      {
        "level_prices": 1.2587056439474198,
        "log_dmd": 1.0719988147046307,
        "valueF": 15.8378587397733,
        "lambda1": 0.000015997886202458523,
        "lambda2": 0.9733476407048465,
        "lambda3": 0.026636361408951224,
        "t": 19,
        "firm_id": 2,
        "demand": 2.9212126311232107
      },
      {
        "level_prices": 1.2586387952175018,
        "log_dmd": 0.6166508245855594,
        "valueF": 15.849449633042957,
        "lambda1": 0.000003409489010092839,
        "lambda2": 0.9680683111041866,
        "lambda3": 0.03192827940680329,
        "t": 20,
        "firm_id": 2,
        "demand": 1.85271257989744
      },
      {
        "level_prices": 1.2586308735756644,
        "log_dmd": 0.839996682827455,
        "valueF": 15.847897834366599,
        "lambda1": 0.0000019177512614867134,
        "lambda2": 0.9687704298246468,
        "lambda3": 0.031227652424091656,
        "t": 21,
        "firm_id": 2,
        "demand": 2.316359293004896
      },
      {
        "level_prices": 1.2586242486877959,
        "log_dmd": 0.3690700285777226,
        "valueF": 15.853416195956049,
        "lambda1": 6.70207442093422e-7,
        "lambda2": 0.9662639638718105,
        "lambda3": 0.03373536592074729,
        "t": 22,
        "firm_id": 2,
        "demand": 1.446388888684984
      },
      {
        "level_prices": 1.2586240826100443,
        "log_dmd": 1.4837287797435823,
        "valueF": 15.844192574432185,
        "lambda1": 6.389330602612643e-7,
        "lambda2": 0.9704502571391366,
        "lambda3": 0.02954910392780318,
        "t": 23,
        "firm_id": 2,
        "demand": 4.409356585097473
      },
      {
        "level_prices": 1.258620990061798,
        "log_dmd": 0.9392877204978262,
        "valueF": 15.872641593642488,
        "lambda1": 5.657007882070769e-8,
        "lambda2": 0.9575371985158672,
        "lambda3": 0.04246274491405405,
        "t": 24,
        "firm_id": 2,
        "demand": 2.5581586452780543
      },
      {
        "level_prices": 1.2586207744905353,
        "log_dmd": 2.0660307750174374,
        "valueF": 15.884552752926139,
        "lambda1": 1.5975490450834057e-8,
        "lambda2": 0.9521310024093289,
        "lambda3": 0.047868981615180724,
        "t": 25,
        "firm_id": 2,
        "demand": 7.8934300562257675
      },
      {
        "level_prices": 1.2586206928473533,
        "log_dmd": 1.1987532434424577,
        "valueF": 15.977802650826451,
        "lambda1": 4.0074998472168373e-10,
        "lambda2": 0.9126471411810004,
        "lambda3": 0.08735285841824961,
        "t": 26,
        "firm_id": 2,
        "demand": 3.315980124517326
      },
      {
        "level_prices": 1.2586206901663335,
        "log_dmd": 0.9984162031771029,
        "valueF": 16.03386786939148,
        "lambda1": 6.417172657499679e-11,
        "lambda2": 0.8910516634157414,
        "lambda3": 0.10894833652008691,
        "t": 27,
        "firm_id": 2,
        "demand": 2.7139800298221233
      },
      {
        "level_prices": 1.2586206897810093,
        "log_dmd": 0.9756321353533597,
        "valueF": 16.074713620057455,
        "lambda1": 1.579768629368054e-11,
        "lambda2": 0.8753184957050083,
        "lambda3": 0.1246815042791941,
        "t": 28,
        "firm_id": 2,
        "demand": 2.652843637312688
      },
      {
        "level_prices": 1.2586206896876464,
        "log_dmd": 1.0542529942382999,
        "valueF": 16.11720654203429,
        "lambda1": 4.076820103718387e-12,
        "lambda2": 0.8589508624727762,
        "lambda3": 0.14104913752314702,
        "t": 29,
        "firm_id": 2,
        "demand": 2.8698305730207854
      },
      {
        "level_prices": 1.3149584653941508,
        "log_dmd": 1.4427104288428445,
        "valueF": 16.192076649640626,
        "lambda1": 8.834685735796102e-13,
        "lambda2": 0.8359247338152429,
        "lambda3": 0.16407526618387366,
        "t": 30,
        "firm_id": 2,
        "demand": 4.232151230203969
      },
      {
        "level_prices": 1.5089838698394193,
        "log_dmd": 0.0031833603694541024,
        "valueF": 16.44576887609111,
        "lambda1": 3.702137018937606e-14,
        "lambda2": 0.7628502308397733,
        "lambda3": 0.23714976916018965,
        "t": 31,
        "firm_id": 2,
        "demand": 1.0031884326419374
      },
      {
        "level_prices": 1.3645885435422536,
        "log_dmd": 0.8436346687224064,
        "valueF": 16.253897558526635,
        "lambda1": 2.0389141577175605e-14,
        "lambda2": 0.8172328861984332,
        "lambda3": 0.18276711380154637,
        "t": 32,
        "firm_id": 2,
        "demand": 2.3248015224853416
      },
      {
        "level_prices": 1.441414620888021,
        "log_dmd": 0.4403661919944118,
        "valueF": 16.349594726543454,
        "lambda1": 2.44124715767639e-15,
        "lambda2": 0.7882983895356878,
        "lambda3": 0.2117016104643098,
        "t": 33,
        "firm_id": 2,
        "demand": 1.5532759115834904
      },
      {
        "level_prices": 1.4319226252971564,
        "log_dmd": 0.03320182529258375,
        "valueF": 16.337771174850406,
        "lambda1": 4.840191827436483e-16,
        "lambda2": 0.7918732969660075,
        "lambda3": 0.20812670303399206,
        "t": 34,
        "firm_id": 2,
        "demand": 1.0337591569327431
      },
      {
        "level_prices": 1.3046985681815522,
        "log_dmd": 0.2355864344178814,
        "valueF": 16.17929657364484,
        "lambda1": 4.2605809349779137e-16,
        "lambda2": 0.8397888509446114,
        "lambda3": 0.1602111490553882,
        "t": 35,
        "firm_id": 2,
        "demand": 1.2656507723173118
      },
      {
        "level_prices": 1.2586206896551762,
        "log_dmd": 0.6362095852246007,
        "valueF": 16.101296198095238,
        "lambda1": 4.458088869672763e-16,
        "lambda2": 0.8650792871991878,
        "lambda3": 0.13492071280081178,
        "t": 36,
        "firm_id": 2,
        "demand": 1.8893060365639642
      },
      {
        "level_prices": 1.2586206896551746,
        "log_dmd": 0.9029182997573675,
        "valueF": 16.097098525126164,
        "lambda1": 2.408939980442068e-16,
        "lambda2": 0.8666961675719275,
        "lambda3": 0.1333038324280722,
        "t": 37,
        "firm_id": 2,
        "demand": 2.466791454087564
      },
      {
        "level_prices": 1.2679724256114513,
        "log_dmd": 0.1917801082372831,
        "valueF": 16.133549244644584,
        "lambda1": 7.277401325688595e-17,
        "lambda2": 0.8536207747697133,
        "lambda3": 0.14637922523028674,
        "t": 38,
        "firm_id": 2,
        "demand": 1.211404109890527
      },
      {
        "level_prices": 1.258620689655173,
        "log_dmd": 0.029259102011028992,
        "valueF": 16.06499050809001,
        "lambda1": 9.989540060331997e-17,
        "lambda2": 0.8790636918969178,
        "lambda3": 0.12093630810308217,
        "t": 39,
        "firm_id": 2,
        "demand": 1.029691355015366
      },
      {
        "level_prices": 1.258620689655174,
        "log_dmd": -0.37083793395401965,
        "valueF": 15.99264069055777,
        "lambda1": 2.0080449835432436e-16,
        "lambda2": 0.9069317513672674,
        "lambda3": 0.09306824863273246,
        "t": 40,
        "firm_id": 2,
        "demand": 0.6901557833145809
      },
      {
        "level_prices": 1.2586206896551775,
        "log_dmd": 0.6365598317236394,
        "valueF": 15.910982937833056,
        "lambda1": 9.457253997259285e-16,
        "lambda2": 0.9401350699819959,
        "lambda3": 0.059864930018003075,
        "t": 41,
        "firm_id": 2,
        "demand": 1.8899678752854652
      },
      {
        "level_prices": 1.2586206896551753,
        "log_dmd": 0.5762267638858402,
        "valueF": 15.909286657634244,
        "lambda1": 5.101113263287471e-16,
        "lambda2": 0.9409049632981303,
        "lambda3": 0.05909503670186913,
        "t": 42,
        "firm_id": 2,
        "demand": 1.7793119842578424
      },
      {
        "level_prices": 1.258620689655174,
        "log_dmd": 0.1540494400705057,
        "valueF": 15.904292551591087,
        "lambda1": 3.1311395035939315e-16,
        "lambda2": 0.943171645928384,
        "lambda3": 0.05682835407161574,
        "t": 43,
        "firm_id": 2,
        "demand": 1.1665485595957883
      },
      {
        "level_prices": 1.2586206896551748,
        "log_dmd": 0.39783719613206275,
        "valueF": 15.879192533162067,
        "lambda1": 4.741370854167674e-16,
        "lambda2": 0.954563830095177,
        "lambda3": 0.045436169904822514,
        "t": 44,
        "firm_id": 2,
        "demand": 1.4886016600703622
      },
      {
        "level_prices": 1.2586206896551746,
        "log_dmd": 1.8943120373170357,
        "valueF": 15.868024612283968,
        "lambda1": 4.2573787870474914e-16,
        "lambda2": 0.9596326316087405,
        "lambda3": 0.04036736839125913,
        "t": 45,
        "firm_id": 2,
        "demand": 6.647973273524054
      },
      {
        "level_prices": 1.2586206896551726,
        "log_dmd": 1.0947805941039652,
        "valueF": 15.93089335945694,
        "lambda1": 1.5547541596179085e-17,
        "lambda2": 0.9310982961398075,
        "lambda3": 0.06890170386019241,
        "t": 46,
        "firm_id": 2,
        "demand": 2.988526911031094
      },
      {
        "level_prices": 1.2586206896551726,
        "log_dmd": 0.5368496296404308,
        "valueF": 15.965637474127709,
        "lambda1": 3.130528879146612e-18,
        "lambda2": 0.9173329834232895,
        "lambda3": 0.0826670165767106,
        "t": 47,
        "firm_id": 2,
        "demand": 1.7106093116351953
      },
      {
        "level_prices": 1.2586206896551724,
        "log_dmd": 0.2523304023441708,
        "valueF": 15.954185735440397,
        "lambda1": 2.0935454090595558e-18,
        "lambda2": 0.9217440205988116,
        "lambda3": 0.07825597940118847,
        "t": 48,
        "firm_id": 2,
        "demand": 1.2870212018653797
      },
      {
        "level_prices": 1.2586206896551724,
        "log_dmd": 0.7946926882650244,
        "valueF": 15.923554815995661,
        "lambda1": 2.579183452984486e-18,
        "lambda2": 0.9344290522053187,
        "lambda3": 0.06557094779468135,
        "t": 49,
        "firm_id": 2,
        "demand": 2.2137605776550946
      },
      {
        "level_prices": 0.6896551724137931,
        "log_dmd": 1.3372632053126494,
        "valueF": 19.774116879259957,
        "lambda1": 0.4,
        "lambda2": 0.4,
        "lambda3": 0.2,
        "t": 0,
        "firm_id": 3,
        "demand": 3.8086058567555923
      },
      {
        "level_prices": 1.9987262005371969,
        "log_dmd": 0.8890165517258476,
        "valueF": 17.28745745271434,
        "lambda1": 0.050934621433147254,
        "lambda2": 0.6003825067193093,
        "lambda3": 0.3486828718475435,
        "t": 1,
        "firm_id": 3,
        "demand": 2.4327360044438278
      },
      {
        "level_prices": 2.4600576666848353,
        "log_dmd": -0.10726176502069795,
        "valueF": 18.520979237460146,
        "lambda1": 1.9835510879643816e-7,
        "lambda2": 0.33322483627008304,
        "lambda3": 0.6667749653748081,
        "t": 2,
        "firm_id": 3,
        "demand": 0.8982905011801373
      },
      {
        "level_prices": 2.5862068965493936,
        "log_dmd": -0.5805769101602359,
        "valueF": 18.94544685992166,
        "lambda1": 8.778255225200251e-13,
        "lambda2": 0.25920216037453353,
        "lambda3": 0.7407978396245886,
        "t": 3,
        "firm_id": 3,
        "demand": 0.559575448665296
      },
      {
        "level_prices": 2.442921313865158,
        "log_dmd": -1.1262477352047502,
        "valueF": 18.484459108841857,
        "lambda1": 6.031036272372981e-17,
        "lambda2": 0.33967898568714805,
        "lambda3": 0.6603210143128518,
        "t": 4,
        "firm_id": 3,
        "demand": 0.32424763965692627
      },
      {
        "level_prices": 1.6379310344827585,
        "log_dmd": -0.4794773877957006,
        "valueF": 16.71778913061083,
        "lambda1": 3.928618321892164e-18,
        "lambda2": 0.6942301407118707,
        "lambda3": 0.3057698592881292,
        "t": 5,
        "firm_id": 3,
        "demand": 0.619106860075387
      },
      {
        "level_prices": 1.335086622313429,
        "log_dmd": 1.0230436311287332,
        "valueF": 16.217148964687418,
        "lambda1": 6.534979879411269e-18,
        "lambda2": 0.8283439993884488,
        "lambda3": 0.17165600061155112,
        "t": 6,
        "firm_id": 3,
        "demand": 2.781648203924607
      },
      {
        "level_prices": 1.4397708569784191,
        "log_dmd": 0.7146764374648795,
        "valueF": 16.347547198458326,
        "lambda1": 6.552569714792466e-19,
        "lambda2": 0.7889174694496863,
        "lambda3": 0.2110825305503136,
        "t": 7,
        "firm_id": 3,
        "demand": 2.0435253670275717
      },
      {
        "level_prices": 1.5221554456818915,
        "log_dmd": -0.10628336577067565,
        "valueF": 16.464781079360048,
        "lambda1": 5.0318656305685713e-20,
        "lambda2": 0.7578895074704564,
        "lambda3": 0.24211049252954356,
        "t": 8,
        "firm_id": 3,
        "demand": 0.8991698180241195
      },
      {
        "level_prices": 1.3408169579289924,
        "log_dmd": 1.0408186305283138,
        "valueF": 16.224286865113164,
        "lambda1": 3.8575577377129094e-20,
        "lambda2": 0.8261858210397302,
        "lambda3": 0.1738141789602698,
        "t": 9,
        "firm_id": 3,
        "demand": 2.8315340460377714
      },
      {
        "level_prices": 1.4544825482002466,
        "log_dmd": 0.2979863085416924,
        "valueF": 16.3671002173208,
        "lambda1": 3.4350335019297714e-21,
        "lambda2": 0.7833767026258811,
        "lambda3": 0.21662329737411892,
        "t": 10,
        "firm_id": 3,
        "demand": 1.347143343396368
      },
      {
        "level_prices": 1.402686034526091,
        "log_dmd": 0.31670367713330994,
        "valueF": 16.301353085629717,
        "lambda1": 9.980019858140096e-22,
        "lambda2": 0.8028844805031605,
        "lambda3": 0.19711551949683945,
        "t": 11,
        "firm_id": 3,
        "demand": 1.3725957801412034
      },
      {
        "level_prices": 1.350056232803321,
        "log_dmd": -0.8216084255649501,
        "valueF": 16.235795619308977,
        "lambda1": 4.262348856829616e-22,
        "lambda2": 0.822706094139009,
        "lambda3": 0.17729390586099106,
        "t": 12,
        "firm_id": 3,
        "demand": 0.4397238223735261
      },
      {
        "level_prices": 1.2586206896551724,
        "log_dmd": 0.9956034610094935,
        "valueF": 15.973566991293778,
        "lambda1": 6.826147943096024e-21,
        "lambda2": 0.9142786529012163,
        "lambda3": 0.08572134709878357,
        "t": 13,
        "firm_id": 3,
        "demand": 2.7063570295431485
      },
      {
        "level_prices": 1.2586206896551724,
        "log_dmd": 0.662810131855098,
        "valueF": 16.006368992155377,
        "lambda1": 1.697085296650486e-21,
        "lambda2": 0.9016438164643423,
        "lambda3": 0.09835618353565763,
        "t": 14,
        "firm_id": 3,
        "demand": 1.9402370025086046
      },
      {
        "level_prices": 1.2586206896551724,
        "log_dmd": 0.09624633552663708,
        "valueF": 16.005980967155104,
        "lambda1": 8.653567331243067e-22,
        "lambda2": 0.901793277847279,
        "lambda3": 0.09820672215272092,
        "t": 15,
        "firm_id": 3,
        "demand": 1.1010302534581369
      },
      {
        "level_prices": 1.2586206896551724,
        "log_dmd": 0.8926008823904499,
        "valueF": 15.951699251482193,
        "lambda1": 1.4971308451187307e-21,
        "lambda2": 0.9227017767505585,
        "lambda3": 0.0772982232494415,
        "t": 16,
        "firm_id": 3,
        "demand": 2.4414713805894745
      },
      {
        "level_prices": 1.2586206896551724,
        "log_dmd": 0.7046652953013532,
        "valueF": 15.971828067063168,
        "lambda1": 4.654031545949251e-22,
        "lambda2": 0.9149484603109099,
        "lambda3": 0.08505153968908999,
        "t": 17,
        "firm_id": 3,
        "demand": 2.023169407277891
      },
      {
        "level_prices": 1.2586206896551724,
        "log_dmd": 0.6097045002609658,
        "valueF": 15.975403299981872,
        "lambda1": 2.1683451380540986e-22,
        "lambda2": 0.9135713344659905,
        "lambda3": 0.08642866553400946,
        "t": 18,
        "firm_id": 3,
        "demand": 1.8398876321289959
      },
      {
        "level_prices": 1.2586206896551724,
        "log_dmd": 0.06356016812400134,
        "valueF": 15.97010713640363,
        "lambda1": 1.2397166812406302e-22,
        "lambda2": 0.9156113368681718,
        "lambda3": 0.08438866313182816,
        "t": 19,
        "firm_id": 3,
        "demand": 1.065623600445549
      },
      {
        "level_prices": 1.2586206896551724,
        "log_dmd": 1.4633299407829865,
        "valueF": 15.922919323799917,
        "lambda1": 2.293197854206092e-22,
        "lambda2": 0.9347174840306159,
        "lambda3": 0.065282515969384,
        "t": 20,
        "firm_id": 3,
        "demand": 4.32032201843774
      },
      {
        "level_prices": 1.2586206896551724,
        "log_dmd": 0.5654595507429923,
        "valueF": 15.988680387479448,
        "lambda1": 2.0882438637350165e-23,
        "lambda2": 0.9084572004350296,
        "lambda3": 0.09154279956497027,
        "t": 21,
        "firm_id": 3,
        "demand": 1.7602565240722323
      },
      {
        "level_prices": 1.2586206896551724,
        "log_dmd": 1.152681469448734,
        "valueF": 15.978835786141065,
        "lambda1": 1.3136453286771157e-23,
        "lambda2": 0.9122491925217773,
        "lambda3": 0.08775080747822268,
        "t": 22,
        "firm_id": 3,
        "demand": 3.1666728717888497
      },
      {
        "level_prices": 1.2586206896551724,
        "log_dmd": 0.7043124289946096,
        "valueF": 16.02980749611051,
        "lambda1": 2.3240252745252467e-24,
        "lambda2": 0.892615657936829,
        "lambda3": 0.10738434206317098,
        "t": 23,
        "firm_id": 3,
        "demand": 2.0224556249035097
      },
      {
        "level_prices": 1.2586206896551724,
        "log_dmd": 0.4536565895774685,
        "valueF": 16.034168637817565,
        "lambda1": 1.0831792011554053e-24,
        "lambda2": 0.890935811869537,
        "lambda3": 0.10906418813046298,
        "t": 24,
        "firm_id": 3,
        "demand": 1.5740573569474978
      },
      {
        "level_prices": 1.2586206896551724,
        "log_dmd": 0.36589185138079244,
        "valueF": 16.010399481680444,
        "lambda1": 8.685182667106387e-25,
        "lambda2": 0.9000913326349371,
        "lambda3": 0.09990866736506293,
        "t": 25,
        "firm_id": 3,
        "demand": 1.4417993056184035
      },
      {
        "level_prices": 1.2586206896551724,
        "log_dmd": 0.9701724565416059,
        "valueF": 15.979854859630645,
        "lambda1": 8.40845305540964e-25,
        "lambda2": 0.9118566607700616,
        "lambda3": 0.08814333922993844,
        "t": 26,
        "firm_id": 3,
        "demand": 2.6383994293627104
      },
      {
        "level_prices": 1.2586206896551724,
        "log_dmd": 0.5900903777610516,
        "valueF": 16.010733901187642,
        "lambda1": 2.207788280919109e-25,
        "lambda2": 0.8999625192801495,
        "lambda3": 0.10003748071985057,
        "t": 27,
        "firm_id": 3,
        "demand": 1.80415146319964
      },
      {
        "level_prices": 1.2586206896551724,
        "log_dmd": 0.9056637924816585,
        "valueF": 16.002635227193558,
        "lambda1": 1.3173814647791882e-25,
        "lambda2": 0.9030820064633407,
        "lambda3": 0.0969179935366592,
        "t": 28,
        "firm_id": 3,
        "demand": 2.4735733175955765
      },
      {
        "level_prices": 1.2586206896551724,
        "log_dmd": 0.6553496366282904,
        "valueF": 16.028770923722686,
        "lambda1": 3.972575204899352e-26,
        "lambda2": 0.8930149300004574,
        "lambda3": 0.10698506999954263,
        "t": 29,
        "firm_id": 3,
        "demand": 1.9258157353990981
      },
      {
        "level_prices": 1.2586206896551724,
        "log_dmd": 1.244106734986886,
        "valueF": 16.027503891009566,
        "lambda1": 2.0586635263096516e-26,
        "lambda2": 0.8935029719085711,
        "lambda3": 0.10649702809142891,
        "t": 30,
        "firm_id": 3,
        "demand": 3.469833933341491
      },
      {
        "level_prices": 1.2586206896551724,
        "log_dmd": 0.9103006030105357,
        "valueF": 16.10033684858479,
        "lambda1": 2.9711162766626867e-27,
        "lambda2": 0.8654488141741269,
        "lambda3": 0.1345511858258731,
        "t": 31,
        "firm_id": 3,
        "demand": 2.485069440473129
      },
      {
        "level_prices": 1.2726905198269691,
        "log_dmd": 0.6634140244989345,
        "valueF": 16.139426262845436,
        "lambda1": 8.830919198108434e-28,
        "lambda2": 0.8518438301950375,
        "lambda3": 0.14815616980496255,
        "t": 32,
        "firm_id": 3,
        "demand": 1.9414090512218978
      },
      {
        "level_prices": 1.2748053189670214,
        "log_dmd": 0.7546731950345587,
        "valueF": 16.14206052839197,
        "lambda1": 4.052958381125539e-28,
        "lambda2": 0.8510473474020308,
        "lambda3": 0.14895265259796925,
        "t": 33,
        "firm_id": 3,
        "demand": 2.126916322948595
      },
      {
        "level_prices": 1.2925675457979455,
        "log_dmd": 0.867895525347888,
        "valueF": 16.164185760658345,
        "lambda1": 1.482763843592595e-28,
        "lambda2": 0.8443576775566178,
        "lambda3": 0.15564232244338222,
        "t": 34,
        "firm_id": 3,
        "demand": 2.381892942021915
      },
      {
        "level_prices": 1.3373973269178827,
        "log_dmd": 0.3637766036233552,
        "valueF": 16.220027256725864,
        "lambda1": 3.530230574513779e-29,
        "lambda2": 0.8274737340179402,
        "lambda3": 0.1725262659820598,
        "t": 35,
        "firm_id": 3,
        "demand": 1.438752766099383
      },
      {
        "level_prices": 1.2919979855746087,
        "log_dmd": 0.7293082984020803,
        "valueF": 16.16347629712368,
        "lambda1": 2.1540683648955084e-29,
        "lambda2": 0.8445721872511214,
        "lambda3": 0.15542781274887857,
        "t": 36,
        "firm_id": 3,
        "demand": 2.073645767399846
      },
      {
        "level_prices": 1.3103185509527404,
        "log_dmd": 1.2036263045028281,
        "valueF": 16.186297014824277,
        "lambda1": 7.234364308118244e-30,
        "lambda2": 0.8376722340567601,
        "lambda3": 0.16232776594323994,
        "t": 37,
        "firm_id": 3,
        "demand": 3.3321785340034094
      },
      {
        "level_prices": 1.439463574964054,
        "log_dmd": -0.149059588714762,
        "valueF": 16.34716443755769,
        "lambda1": 6.0498187837399814e-31,
        "lambda2": 0.7890331990395121,
        "lambda3": 0.21096680096048787,
        "t": 38,
        "firm_id": 3,
        "demand": 0.8615177766323286
      },
      {
        "level_prices": 1.2652767461288394,
        "log_dmd": -0.16904987076447808,
        "valueF": 16.130191414838887,
        "lambda1": 9.502123408415065e-31,
        "lambda2": 0.8546360306787487,
        "lambda3": 0.14536396932125134,
        "t": 39,
        "firm_id": 3,
        "demand": 0.8444667881315826
      },
      {
        "level_prices": 1.2586206896551724,
        "log_dmd": 0.47041822148938095,
        "valueF": 16.01921947013782,
        "lambda1": 2.932916043250452e-30,
        "lambda2": 0.8966940059900091,
        "lambda3": 0.10330599400999084,
        "t": 40,
        "firm_id": 3,
        "demand": 1.6006634851182209
      },
      {
        "level_prices": 1.2586206896551724,
        "log_dmd": 0.6862124467246549,
        "valueF": 15.99829171784683,
        "lambda1": 2.2668374430515486e-30,
        "lambda2": 0.9047550608111385,
        "lambda3": 0.0952449391888615,
        "t": 41,
        "firm_id": 3,
        "demand": 1.986178511890227
      },
      {
        "level_prices": 1.2586206896551724,
        "log_dmd": 0.3824363920619763,
        "valueF": 16.00033041884548,
        "lambda1": 1.0988927232572348e-30,
        "lambda2": 0.9039697839077471,
        "lambda3": 0.09603021609225294,
        "t": 42,
        "firm_id": 3,
        "demand": 1.4658516315923733
      },
      {
        "level_prices": 1.2586206896551724,
        "log_dmd": 0.44379620186859886,
        "valueF": 15.972396352056219,
        "lambda1": 1.0260919409686816e-30,
        "lambda2": 0.9147295654951013,
        "lambda3": 0.08527043450489875,
        "t": 43,
        "firm_id": 3,
        "demand": 1.5586128108738209
      },
      {
        "level_prices": 1.2586206896551724,
        "log_dmd": -0.007306745545088589,
        "valueF": 15.952514445177666,
        "lambda1": 8.385087517491464e-31,
        "lambda2": 0.9223877764232572,
        "lambda3": 0.07761222357674286,
        "t": 44,
        "firm_id": 3,
        "demand": 0.9927198838226645
      },
      {
        "level_prices": 1.2586206896551724,
        "log_dmd": 0.36538308778976436,
        "valueF": 15.907158290861489,
        "lambda1": 1.8024547240063933e-30,
        "lambda2": 0.9418709684155468,
        "lambda3": 0.05812903158445318,
        "t": 45,
        "firm_id": 3,
        "demand": 1.441065957192443
      },
      {
        "level_prices": 1.2586206896551724,
        "log_dmd": 0.9370648804333087,
        "valueF": 15.891436993208824,
        "lambda1": 1.7374280370324207e-30,
        "lambda2": 0.9490064180848995,
        "lambda3": 0.05099358191510043,
        "t": 46,
        "firm_id": 3,
        "demand": 2.5524785830236865
      },
      {
        "level_prices": 1.2586206896551724,
        "log_dmd": 0.8641587035431194,
        "valueF": 15.905475961950016,
        "lambda1": 4.924378649587885e-31,
        "lambda2": 0.9426345296393293,
        "lambda3": 0.05736547036067073,
        "t": 47,
        "firm_id": 3,
        "demand": 2.37300884202684
      },
      {
        "level_prices": 1.2586206896551724,
        "log_dmd": 1.0571341451602945,
        "valueF": 15.916757018959991,
        "lambda1": 1.63064573137754e-31,
        "lambda2": 0.9375143788546926,
        "lambda3": 0.06248562114530736,
        "t": 48,
        "firm_id": 3,
        "demand": 2.8781109107456038
      },
      {
        "level_prices": 1.2586206896551724,
        "log_dmd": 0.11035196856273521,
        "valueF": 15.94282173264832,
        "lambda1": 3.565105336362307e-32,
        "lambda2": 0.9261212632295853,
        "lambda3": 0.07387873677041466,
        "t": 49,
        "firm_id": 3,
        "demand": 1.1166710343983632
      },
      {
        "level_prices": 0.6896551724137931,
        "log_dmd": 2.411578941122104,
        "valueF": 19.774116879259957,
        "lambda1": 0.4,
        "lambda2": 0.4,
        "lambda3": 0.2,
        "t": 0,
        "firm_id": 4,
        "demand": 11.151554900265758
      },
      {
        "level_prices": 0.6896551724137931,
        "log_dmd": 2.2032028710632443,
        "valueF": 22.827711062621674,
        "lambda1": 0.7257547449810619,
        "lambda2": 0.21741581794690817,
        "lambda3": 0.056829437072029955,
        "t": 1,
        "firm_id": 4,
        "demand": 9.05396579450058
      },
      {
        "level_prices": 0.6896551724137931,
        "log_dmd": 1.1671571661991362,
        "valueF": 23.84817294550057,
        "lambda1": 0.8249824263078381,
        "lambda2": 0.15093825409713774,
        "lambda3": 0.024079319595024185,
        "t": 2,
        "firm_id": 4,
        "demand": 3.2128460562162333
      },
      {
        "level_prices": 2.206896551724138,
        "log_dmd": -0.08960572643847048,
        "valueF": 17.40101803540755,
        "lambda1": 0.17640970972085576,
        "lambda2": 0.6805014218178258,
        "lambda3": 0.14308886846131838,
        "t": 3,
        "firm_id": 4,
        "demand": 0.9142915952200261
      },
      {
        "level_prices": 1.3772303778648096,
        "log_dmd": 0.6802331944397638,
        "valueF": 16.269357878503534,
        "lambda1": 0.00004309539746834317,
        "lambda2": 0.8126009620615027,
        "lambda3": 0.18735594254102905,
        "t": 4,
        "firm_id": 4,
        "demand": 1.9743380832159363
      },
      {
        "level_prices": 1.415306533663707,
        "log_dmd": 1.3581825539802106,
        "valueF": 16.317024392277627,
        "lambda1": 0.000007395791732048146,
        "lambda2": 0.7981534928784754,
        "lambda3": 0.2018391113297926,
        "t": 5,
        "firm_id": 4,
        "demand": 3.8891186116676164
      },
      {
        "level_prices": 1.637931261275556,
        "log_dmd": 0.29448629478916927,
        "valueF": 16.757991631636575,
        "lambda1": 8.541546918937441e-8,
        "lambda2": 0.6848432857947857,
        "lambda3": 0.3151566287897451,
        "t": 6,
        "firm_id": 4,
        "demand": 1.3424365648675243
      },
      {
        "level_prices": 1.6379310438732875,
        "log_dmd": 0.19743736015808871,
        "valueF": 16.767098623501465,
        "lambda1": 3.536692744342281e-9,
        "lambda2": 0.6827168292039288,
        "lambda3": 0.3172831672593785,
        "t": 7,
        "firm_id": 4,
        "demand": 1.2182767499164637
      },
      {
        "level_prices": 1.6379310351049652,
        "log_dmd": 0.44880226316958055,
        "valueF": 16.688650392000262,
        "lambda1": 2.3433764170466955e-10,
        "lambda2": 0.7010337688877698,
        "lambda3": 0.2989662308778925,
        "t": 8,
        "firm_id": 4,
        "demand": 1.566434884704585
      },
      {
        "level_prices": 1.6379310344949438,
        "log_dmd": -0.180679706942946,
        "valueF": 16.838738863250992,
        "lambda1": 4.589296184596648e-12,
        "lambda2": 0.6659894882996317,
        "lambda3": 0.334010511695779,
        "t": 9,
        "firm_id": 4,
        "demand": 0.8347026653536883
      },
      {
        "level_prices": 1.5182418428355247,
        "log_dmd": 0.6456101228826538,
        "valueF": 16.459132080802146,
        "lambda1": 1.879057491933269e-12,
        "lambda2": 0.759363461792976,
        "lambda3": 0.2406365382051448,
        "t": 10,
        "firm_id": 4,
        "demand": 1.9071502703964522
      },
      {
        "level_prices": 1.6254015310649115,
        "log_dmd": 0.4258780400411311,
        "valueF": 16.61380922796422,
        "lambda1": 7.179588515174284e-14,
        "lambda2": 0.7190046181705014,
        "lambda3": 0.2809953818294268,
        "t": 11,
        "firm_id": 4,
        "demand": 1.5309340510931484
      },
      {
        "level_prices": 1.6379310344827633,
        "log_dmd": -0.18617591483314877,
        "valueF": 16.724119591584813,
        "lambda1": 1.850650644402827e-15,
        "lambda2": 0.692752036179571,
        "lambda3": 0.30724796382042713,
        "t": 12,
        "firm_id": 4,
        "demand": 0.8301275503870855
      },
      {
        "level_prices": 1.4580597140062472,
        "log_dmd": 0.7135724252060565,
        "valueF": 16.372263593688448,
        "lambda1": 7.691610532165384e-16,
        "lambda2": 0.782029458361285,
        "lambda3": 0.21797054163871418,
        "t": 13,
        "firm_id": 4,
        "demand": 2.0412705348813467
      },
      {
        "level_prices": 1.5518334790823993,
        "log_dmd": -0.1444947255052108,
        "valueF": 16.50761914295822,
        "lambda1": 4.765959510596272e-17,
        "lambda2": 0.7467120663196158,
        "lambda3": 0.25328793368038416,
        "t": 14,
        "firm_id": 4,
        "demand": 0.8654594772525712
      },
      {
        "level_prices": 1.3530749210848956,
        "log_dmd": 0.6039415111254862,
        "valueF": 16.239555799831688,
        "lambda1": 3.3899345224918494e-17,
        "lambda2": 0.821569185565429,
        "lambda3": 0.178430814434571,
        "t": 15,
        "firm_id": 4,
        "demand": 1.8293148743006733
      },
      {
        "level_prices": 1.363136024389501,
        "log_dmd": 0.8648771692625137,
        "valueF": 16.252088251398114,
        "lambda1": 9.23664817365131e-18,
        "lambda2": 0.8177799388662919,
        "lambda3": 0.18222006113370817,
        "t": 16,
        "firm_id": 4,
        "demand": 2.374714380143625
      },
      {
        "level_prices": 1.444927899418082,
        "log_dmd": 0.9999989106560583,
        "valueF": 16.353970985457064,
        "lambda1": 1.0550554246079551e-18,
        "lambda2": 0.7869752067126704,
        "lambda3": 0.21302479328732962,
        "t": 17,
        "firm_id": 4,
        "demand": 2.7182788673168163
      },
      {
        "level_prices": 1.6379310344827585,
        "log_dmd": 0.6176292527904284,
        "valueF": 16.63395685289244,
        "lambda1": 2.743446592877956e-20,
        "lambda2": 0.7138042080914605,
        "lambda3": 0.28619579190853955,
        "t": 18,
        "firm_id": 4,
        "demand": 1.8545262132514546
      },
      {
        "level_prices": 1.6395093040668505,
        "log_dmd": 0.1219519865537756,
        "valueF": 16.94057329872656,
        "lambda1": 2.3646126181171894e-22,
        "lambda2": 0.6422627296371601,
        "lambda3": 0.35773727036283987,
        "t": 19,
        "firm_id": 4,
        "demand": 1.1296998596946912
      },
      {
        "level_prices": 1.6379310344827585,
        "log_dmd": 0.09189161807689039,
        "valueF": 16.787328802617715,
        "lambda1": 2.2304122971382805e-23,
        "lambda2": 0.6779932655790027,
        "lambda3": 0.32200673442099725,
        "t": 20,
        "firm_id": 4,
        "demand": 1.0962460023920397
      },
      {
        "level_prices": 1.6280141314399617,
        "log_dmd": -0.2521723056898276,
        "valueF": 16.617580324907273,
        "lambda1": 2.46109653049448e-24,
        "lambda2": 0.718020651795339,
        "lambda3": 0.281979348204661,
        "t": 21,
        "firm_id": 4,
        "demand": 0.7771108259154326
      },
      {
        "level_prices": 1.377140133199771,
        "log_dmd": 1.1857631970681366,
        "valueF": 16.26953224437603,
        "lambda1": 1.5080687797215929e-24,
        "lambda2": 0.8125056641195667,
        "lambda3": 0.1874943358804333,
        "t": 22,
        "firm_id": 4,
        "demand": 3.2731839528393505
      },
      {
        "level_prices": 1.5673398247243762,
        "log_dmd": -0.13180817246743926,
        "valueF": 16.53000141511965,
        "lambda1": 5.428832743169842e-26,
        "lambda2": 0.7408720140648453,
        "lambda3": 0.25912798593515485,
        "t": 23,
        "firm_id": 4,
        "demand": 0.8765091175348836
      },
      {
        "level_prices": 1.370811725485542,
        "log_dmd": 0.03951942524733132,
        "valueF": 16.261649365056538,
        "lambda1": 3.226027634810331e-26,
        "lambda2": 0.814889090401549,
        "lambda3": 0.185110909598451,
        "t": 24,
        "firm_id": 4,
        "demand": 1.0403107069814683
      },
      {
        "level_prices": 1.2586206896551724,
        "log_dmd": 0.32828037653269126,
        "valueF": 16.11945140493932,
        "lambda1": 4.061679628080177e-26,
        "lambda2": 0.8580861751045797,
        "lambda3": 0.14191382489542012,
        "t": 25,
        "firm_id": 4,
        "demand": 1.3885782424687447
      },
      {
        "level_prices": 1.2586206896551724,
        "log_dmd": 1.1591333318138404,
        "valueF": 16.072949635940535,
        "lambda1": 4.29082875325598e-26,
        "lambda2": 0.8759979557853959,
        "lambda3": 0.12400204421460408,
        "t": 26,
        "firm_id": 4,
        "demand": 3.1871698600851524
      },
      {
        "level_prices": 1.2796807807161843,
        "log_dmd": 0.9007003424190072,
        "valueF": 16.14813356893091,
        "lambda1": 7.418839988851533e-27,
        "lambda2": 0.8492111345354629,
        "lambda3": 0.150788865464537,
        "t": 27,
        "firm_id": 4,
        "demand": 2.4613262788829102
      },
      {
        "level_prices": 1.3249464796306687,
        "log_dmd": 0.9227272118673115,
        "valueF": 16.20451805899041,
        "lambda1": 1.8485825536893255e-27,
        "lambda2": 0.8321630141650728,
        "lambda3": 0.1678369858349272,
        "t": 28,
        "firm_id": 4,
        "demand": 2.5161430966103895
      },
      {
        "level_prices": 1.3982856060944973,
        "log_dmd": 1.3542797739233883,
        "valueF": 16.2958717627051,
        "lambda1": 2.7385682334840317e-28,
        "lambda2": 0.8045417847176568,
        "lambda3": 0.19545821528234322,
        "t": 29,
        "firm_id": 4,
        "demand": 3.8739698175456514
      },
      {
        "level_prices": 1.6379310344827585,
        "log_dmd": 0.2201503825129357,
        "valueF": 16.68934525263677,
        "lambda1": 4.1999945549099626e-30,
        "lambda2": 0.7008715251422369,
        "lambda3": 0.29912847485776306,
        "t": 30,
        "firm_id": 4,
        "demand": 1.246264132828094
      },
      {
        "level_prices": 1.6379310344827585,
        "log_dmd": 0.010976468690108543,
        "valueF": 16.63304358575919,
        "lambda1": 2.4904991962048825e-31,
        "lambda2": 0.7140174475845087,
        "lambda3": 0.28598255241549136,
        "t": 31,
        "firm_id": 4,
        "demand": 1.011036931141462
      },
      {
        "level_prices": 1.5010645061209777,
        "log_dmd": 1.2914492819526993,
        "valueF": 16.434337855644284,
        "lambda1": 4.0185494690834013e-32,
        "lambda2": 0.7658328483440473,
        "lambda3": 0.23416715165595264,
        "t": 32,
        "firm_id": 4,
        "demand": 3.638055304775291
      },
      {
        "level_prices": 1.7110443214655833,
        "log_dmd": -0.2306772291042083,
        "valueF": 17.065791702620267,
        "lambda1": 1.43686758997463e-34,
        "lambda2": 0.6153209698376373,
        "lambda3": 0.3846790301623627,
        "t": 33,
        "firm_id": 4,
        "demand": 0.793995703384475
      },
      {
        "level_prices": 1.6170539408713445,
        "log_dmd": 0.10654247012135604,
        "valueF": 16.601760094021678,
        "lambda1": 3.8281827390279615e-35,
        "lambda2": 0.7221485157757273,
        "lambda3": 0.27785148422427264,
        "t": 34,
        "firm_id": 4,
        "demand": 1.112425170274433
      },
      {
        "level_prices": 1.5207170138214707,
        "log_dmd": 0.4664768728373619,
        "valueF": 16.462704808513973,
        "lambda1": 4.894616697837346e-36,
        "lambda2": 0.7584312545347707,
        "lambda3": 0.24156874546522938,
        "t": 35,
        "firm_id": 4,
        "demand": 1.594367128467964
      },
      {
        "level_prices": 1.5515007389407498,
        "log_dmd": 0.16070845230367153,
        "valueF": 16.50713885697844,
        "lambda1": 3.764542582231488e-37,
        "lambda2": 0.7468373840353019,
        "lambda3": 0.2531626159646982,
        "t": 36,
        "firm_id": 4,
        "demand": 1.1743425420365967
      },
      {
        "level_prices": 1.4662036418665805,
        "log_dmd": 1.117835152332258,
        "valueF": 16.384018755996948,
        "lambda1": 7.414995893672296e-38,
        "lambda2": 0.7789622647515476,
        "lambda3": 0.2210377352484523,
        "t": 37,
        "firm_id": 4,
        "demand": 3.0582264374591923
      },
      {
        "level_prices": 1.6379310344827585,
        "log_dmd": 0.10757762256536663,
        "valueF": 16.787142743219427,
        "lambda1": 9.164791499507174e-40,
        "lambda2": 0.6780367087407283,
        "lambda3": 0.32196329125927164,
        "t": 38,
        "firm_id": 4,
        "demand": 1.1135772961182837
      },
      {
        "level_prices": 1.636256271571546,
        "log_dmd": 0.4091277119738105,
        "valueF": 16.62947724942303,
        "lambda1": 9.37629709548408e-41,
        "lambda2": 0.7149164691483786,
        "lambda3": 0.2850835308516214,
        "t": 39,
        "firm_id": 4,
        "demand": 1.5055039790931848
      },
      {
        "level_prices": 1.6379310344827585,
        "log_dmd": 0.34546701238909694,
        "valueF": 16.737730249817844,
        "lambda1": 2.2794018792166814e-42,
        "lambda2": 0.6895740724097102,
        "lambda3": 0.3104259275902897,
        "t": 40,
        "firm_id": 4,
        "demand": 1.4126494904553009
      },
      {
        "level_prices": 1.6379310344827585,
        "log_dmd": 0.3032621618206056,
        "valueF": 16.79349017327,
        "lambda1": 7.3696341889557935e-44,
        "lambda2": 0.6765546420847968,
        "lambda3": 0.3234453579152033,
        "t": 41,
        "firm_id": 4,
        "demand": 1.3542694556537687
      },
      {
        "level_prices": 1.6379310344827585,
        "log_dmd": 1.0196025995269924,
        "valueF": 16.810880208632362,
        "lambda1": 2.9237644436019435e-45,
        "lambda2": 0.6724942284448415,
        "lambda3": 0.3275057715551584,
        "t": 42,
        "firm_id": 4,
        "demand": 2.7720929140057047
      },
      {
        "level_prices": 2.0211847021576057,
        "log_dmd": 0.17548708939190105,
        "valueF": 17.630828112735422,
        "lambda1": 3.3478304380361474e-48,
        "lambda2": 0.49851485243414845,
        "lambda3": 0.5014851475658515,
        "t": 43,
        "firm_id": 4,
        "demand": 1.1918266013457608
      },
      {
        "level_prices": 2.2029032267943354,
        "log_dmd": -0.23387302063566284,
        "valueF": 17.987709027594814,
        "lambda1": 1.381016940345136e-51,
        "lambda2": 0.4300754080904451,
        "lambda3": 0.5699245919095549,
        "t": 44,
        "firm_id": 4,
        "demand": 0.7914623088960606
      },
      {
        "level_prices": 2.1109727094688324,
        "log_dmd": -0.4073750286283453,
        "valueF": 17.80716470944008,
        "lambda1": 9.218456281671912e-55,
        "lambda2": 0.4646985899403099,
        "lambda3": 0.5353014100596901,
        "t": 45,
        "firm_id": 4,
        "demand": 0.6653946014782688
      },
      {
        "level_prices": 1.791943722772261,
        "log_dmd": 0.5570666598245153,
        "valueF": 17.20740198351408,
        "lambda1": 8.113200921530656e-57,
        "lambda2": 0.5848523641507067,
        "lambda3": 0.4151476358492933,
        "t": 46,
        "firm_id": 4,
        "demand": 1.7455447067750742
      },
      {
        "level_prices": 2.10159114277641,
        "log_dmd": 0.2229965744667135,
        "valueF": 17.788740049044563,
        "lambda1": 9.343715386598414e-60,
        "lambda2": 0.4682319072660275,
        "lambda3": 0.5317680927339726,
        "t": 47,
        "firm_id": 4,
        "demand": 1.2498162924413394
      },
      {
        "level_prices": 2.38734610936094,
        "log_dmd": -0.3339903794564322,
        "valueF": 18.366708662170577,
        "lambda1": 7.61520075107574e-64,
        "lambda2": 0.3606099068640614,
        "lambda3": 0.6393900931359386,
        "t": 48,
        "firm_id": 4,
        "demand": 0.7160606710869546
      },
      {
        "level_prices": 2.309124737725546,
        "log_dmd": -0.7829705401270134,
        "valueF": 18.205819300373257,
        "lambda1": 8.535851826752647e-68,
        "lambda2": 0.3900699039734957,
        "lambda3": 0.6099300960265043,
        "t": 49,
        "firm_id": 4,
        "demand": 0.4570463183652363
      },
      {
        "level_prices": 0.6896551724137931,
        "log_dmd": 1.9524228470605713,
        "valueF": 19.774116879259957,
        "lambda1": 0.4,
        "lambda2": 0.4,
        "lambda3": 0.2,
        "t": 0,
        "firm_id": 5,
        "demand": 7.0457376621630114
      },
      {
        "level_prices": 1.0514460742900702,
        "log_dmd": 0.254985524249981,
        "valueF": 19.017342427102175,
        "lambda1": 0.3368671613207316,
        "lambda2": 0.4848601209893044,
        "lambda3": 0.17827271768996397,
        "t": 1,
        "firm_id": 5,
        "demand": 1.2904429406082627
      },
      {
        "level_prices": 0.6896551724137931,
        "log_dmd": 1.6206118247026078,
        "valueF": 19.648749275083922,
        "lambda1": 0.4057083392074482,
        "lambda2": 0.4422812552492412,
        "lambda3": 0.15201040554331058,
        "t": 2,
        "firm_id": 5,
        "demand": 5.05618286799726
      },
      {
        "level_prices": 2.160533039019682,
        "log_dmd": 0.8252926211719595,
        "valueF": 17.331391220034238,
        "lambda1": 0.13412635150370636,
        "lambda2": 0.6542720548366892,
        "lambda3": 0.2116015936596044,
        "t": 3,
        "firm_id": 5,
        "demand": 2.2825485896581297
      },
      {
        "level_prices": 2.229804678143073,
        "log_dmd": -0.3304773911540471,
        "valueF": 18.04266997991661,
        "lambda1": 5.187896718955174e-8,
        "lambda2": 0.4199437445263814,
        "lambda3": 0.5800562035946515,
        "t": 4,
        "firm_id": 5,
        "demand": 0.7185806075103077
      },
      {
        "level_prices": 2.0531033141616173,
        "log_dmd": -0.15328007377923591,
        "valueF": 17.69351376052517,
        "lambda1": 5.2369342474909215e-11,
        "lambda2": 0.4864935570564356,
        "lambda3": 0.5135064428911951,
        "t": 5,
        "firm_id": 5,
        "demand": 0.8578894158304043
      },
      {
        "level_prices": 1.9479157414616914,
        "log_dmd": 0.26414336478552985,
        "valueF": 17.4939433003044,
        "lambda1": 1.4972636087173975e-13,
        "lambda2": 0.5261096558134286,
        "lambda3": 0.4738903441864216,
        "t": 6,
        "firm_id": 5,
        "demand": 1.3023148890127496
      },
      {
        "level_prices": 2.1518172678968597,
        "log_dmd": 0.0025244612459016935,
        "valueF": 17.88738021530533,
        "lambda1": 1.0109606983006454e-16,
        "lambda2": 0.44931557442845543,
        "lambda3": 0.5506844255715444,
        "t": 7,
        "firm_id": 5,
        "demand": 1.0025276503812444
      },
      {
        "level_prices": 2.264993144310018,
        "log_dmd": 0.6185736087157007,
        "valueF": 18.115047382785068,
        "lambda1": 2.130894884505517e-20,
        "lambda2": 0.4066908937014217,
        "lambda3": 0.5933091062985782,
        "t": 8,
        "firm_id": 5,
        "demand": 1.8562783732706436
      },
      {
        "level_prices": 2.74239925810602,
        "log_dmd": -1.0634116884951528,
        "valueF": 19.56701281765206,
        "lambda1": 3.510882060112565e-27,
        "lambda2": 0.15546001967435613,
        "lambda3": 0.844539980325644,
        "t": 9,
        "firm_id": 5,
        "demand": 0.34527582504353677
      },
      {
        "level_prices": 2.330249579783198,
        "log_dmd": 0.4499671434947737,
        "valueF": 18.24926985980051,
        "lambda1": 4.612481155874021e-30,
        "lambda2": 0.38211379462710715,
        "lambda3": 0.6178862053728928,
        "t": 10,
        "firm_id": 5,
        "demand": 1.5682606570791762
      },
      {
        "level_prices": 2.718503519603778,
        "log_dmd": 0.49915623280809673,
        "valueF": 19.5122603450068,
        "lambda1": 1.0172812340364292e-36,
        "lambda2": 0.16445971339597962,
        "lambda3": 0.8355402866040205,
        "t": 11,
        "firm_id": 5,
        "demand": 1.6473307205161296
      },
      {
        "level_prices": 2.889489641649825,
        "log_dmd": 0.20054945833603438,
        "valueF": 20.380519282585325,
        "lambda1": 7.252545350475133e-47,
        "lambda2": 0.028633771326689254,
        "lambda3": 0.9713662286733107,
        "t": 12,
        "firm_id": 5,
        "demand": 1.2220740524945415
      },
      {
        "level_prices": 2.9493092509080023,
        "log_dmd": -0.6720526903124541,
        "valueF": 20.53078104283356,
        "lambda1": 5.725590547843877e-57,
        "lambda2": 0.006104308099583474,
        "lambda3": 0.9938956919004165,
        "t": 13,
        "firm_id": 5,
        "demand": 0.510659275860509
      },
      {
        "level_prices": 2.9452612871331185,
        "log_dmd": -0.8152656304765018,
        "valueF": 20.520612902824322,
        "lambda1": 5.749060776324752e-63,
        "lambda2": 0.007628865884929387,
        "lambda3": 0.9923711341150705,
        "t": 14,
        "firm_id": 5,
        "demand": 0.44252176448073893
      },
      {
        "level_prices": 2.9310185229820895,
        "log_dmd": -0.7327536045226566,
        "valueF": 20.484836293136063,
        "lambda1": 3.4556725650575784e-68,
        "lambda2": 0.012993023811940247,
        "lambda3": 0.9870069761880598,
        "t": 15,
        "firm_id": 5,
        "demand": 0.48058382863818333
      },
      {
        "level_prices": 2.915707979421007,
        "log_dmd": 0.41793344406477384,
        "valueF": 20.446377512572713,
        "lambda1": 9.441345871379424e-74,
        "lambda2": 0.018759332425854456,
        "lambda3": 0.9812406675741456,
        "t": 16,
        "firm_id": 5,
        "demand": 1.5188195845803218
      },
      {
        "level_prices": 2.9591600562692317,
        "log_dmd": -1.093506272279793,
        "valueF": 20.555525426352563,
        "lambda1": 3.386462102631381e-85,
        "lambda2": 0.0023942645219776322,
        "lambda3": 0.9976057354780223,
        "t": 17,
        "firm_id": 5,
        "demand": 0.3350396914350605
      },
      {
        "level_prices": 2.94595224476833,
        "log_dmd": 0.05881644364936267,
        "valueF": 20.522348529516158,
        "lambda1": 4.895813654572461e-89,
        "lambda2": 0.007368635087252064,
        "lambda3": 0.992631364912748,
        "t": 18,
        "firm_id": 5,
        "demand": 1.060580546577344
      },
      {
        "level_prices": 2.9604249856618146,
        "log_dmd": -0.20558668020026177,
        "valueF": 20.558702821181384,
        "lambda1": 7.5087290422239e-99,
        "lambda2": 0.001917862542952857,
        "lambda3": 0.9980821374570471,
        "t": 19,
        "firm_id": 5,
        "demand": 0.814169519131486
      },
      {
        "level_prices": 2.9632247135136933,
        "log_dmd": 0.027691190407504784,
        "valueF": 20.565735498934654,
        "lambda1": 2.205069816438111e-107,
        "lambda2": 0.0008634195857518682,
        "lambda3": 0.9991365804142481,
        "t": 20,
        "firm_id": 5,
        "demand": 1.0280781549997842
      },
      {
        "level_prices": 2.9648980365600734,
        "log_dmd": -0.6501073794008987,
        "valueF": 20.56993874381591,
        "lambda1": 3.614856398023645e-117,
        "lambda2": 0.00023320700984225915,
        "lambda3": 0.9997667929901578,
        "t": 21,
        "firm_id": 5,
        "demand": 0.5219897228078396
      },
      {
        "level_prices": 2.9647897142200135,
        "log_dmd": -0.19128417351020544,
        "valueF": 20.569666647329598,
        "lambda1": 2.2019708534404533e-123,
        "lambda2": 0.0002740037353193406,
        "lambda3": 0.9997259962646806,
        "t": 22,
        "firm_id": 5,
        "demand": 0.8258978565075611
      },
      {
        "level_prices": 2.9652018328829772,
        "log_dmd": -0.8023212800762081,
        "valueF": 20.570701854287584,
        "lambda1": 5.0447157238421745e-132,
        "lambda2": 0.00011879021290456764,
        "lambda3": 0.9998812097870955,
        "t": 23,
        "firm_id": 5,
        "demand": 0.4482871553778897
      },
      {
        "level_prices": 2.9650014833495586,
        "log_dmd": -1.5619554125449189,
        "valueF": 20.5701985933347,
        "lambda1": 1.951209386369745e-137,
        "lambda2": 0.00019424653068576695,
        "lambda3": 0.9998057534693142,
        "t": 24,
        "firm_id": 5,
        "demand": 0.20972556997142627
      },
      {
        "level_prices": 2.9611255832753223,
        "log_dmd": -0.5820879282034113,
        "valueF": 20.56046266267712,
        "lambda1": 7.841088578344302e-139,
        "lambda2": 0.0016540011040992058,
        "lambda3": 0.9983459988959007,
        "t": 25,
        "firm_id": 5,
        "demand": 0.5587305585487028
      },
      {
        "level_prices": 2.9610486490056993,
        "log_dmd": -0.5063130389648725,
        "valueF": 20.560269410348656,
        "lambda1": 2.211555941377529e-145,
        "lambda2": 0.0016829763485027589,
        "lambda3": 0.9983170236514972,
        "t": 26,
        "firm_id": 5,
        "demand": 0.6027136691066899
      },
      {
        "level_prices": 2.961658786817163,
        "log_dmd": -0.49884648263333187,
        "valueF": 20.561802024531225,
        "lambda1": 2.486220757224126e-152,
        "lambda2": 0.0014531841857436903,
        "lambda3": 0.9985468158142563,
        "t": 27,
        "firm_id": 5,
        "demand": 0.6072307070427814
      },
      {
        "level_prices": 2.9622414504805596,
        "log_dmd": 0.6257370590922562,
        "valueF": 20.563265625995363,
        "lambda1": 2.528113617342553e-159,
        "lambda2": 0.001233739429399665,
        "lambda3": 0.9987662605706004,
        "t": 28,
        "firm_id": 5,
        "demand": 1.869623472694431
      },
      {
        "level_prices": 2.9652753697416108,
        "log_dmd": 0.2593116250995515,
        "valueF": 20.570886572609194,
        "lambda1": 2.9259581613044137e-172,
        "lambda2": 0.00009109451289965358,
        "lambda3": 0.9999089054871003,
        "t": 29,
        "firm_id": 5,
        "demand": 1.2960376197576262
      },
      {
        "level_prices": 2.9654778977116134,
        "log_dmd": 0.40330786830991183,
        "valueF": 20.571395305609116,
        "lambda1": 2.7564325847331804e-183,
        "lambda2": 0.000014817744976701878,
        "lambda3": 0.9999851822550233,
        "t": 30,
        "firm_id": 5,
        "demand": 1.4967676280568567
      },
      {
        "level_prices": 2.965512563656364,
        "log_dmd": -1.101880162715371,
        "valueF": 20.571482383507874,
        "lambda1": 4.4811864986779945e-195,
        "lambda2": 0.0000017617398108415416,
        "lambda3": 0.9999982382601892,
        "t": 31,
        "firm_id": 5,
        "demand": 0.33224581988053675
      },
      {
        "level_prices": 2.9655025737902014,
        "log_dmd": -0.06964807483874588,
        "valueF": 20.571457289815474,
        "lambda1": 6.619905480362683e-199,
        "lambda2": 0.000005524156937101907,
        "lambda3": 0.9999944758430629,
        "t": 32,
        "firm_id": 5,
        "demand": 0.9327220104972471
      },
      {
        "level_prices": 2.9655123654292845,
        "log_dmd": 0.11849348348167588,
        "valueF": 20.571481885578347,
        "lambda1": 3.407561938708884e-208,
        "lambda2": 0.0000018363967627905805,
        "lambda3": 0.9999981636032372,
        "t": 33,
        "firm_id": 5,
        "demand": 1.1257995377847645
      },
      {
        "level_prices": 2.965516164642299,
        "log_dmd": 0.35931886516690814,
        "valueF": 20.571491428877646,
        "lambda1": 1.774840949609121e-218,
        "lambda2": 4.0552432887654333e-7,
        "lambda3": 0.9999995944756712,
        "t": 34,
        "firm_id": 5,
        "demand": 1.4323534563860747
      },
      {
        "level_prices": 2.965517100525184,
        "log_dmd": 0.933575401328623,
        "valueF": 20.57149377973569,
        "lambda1": 4.925781034910101e-230,
        "lambda2": 5.3048956650775413e-8,
        "lambda3": 0.9999999469510434,
        "t": 35,
        "firm_id": 5,
        "demand": 2.5435872843655547
      },
      {
        "level_prices": 2.9655172360922384,
        "log_dmd": -1.0663811766205717,
        "valueF": 20.57149412026858,
        "lambda1": 1.2571503960405052e-244,
        "lambda2": 1.991234652326258e-9,
        "lambda3": 0.9999999980087654,
        "t": 36,
        "firm_id": 5,
        "demand": 0.3442520533722757
      },
      {
        "level_prices": 2.965517226032383,
        "log_dmd": -0.3285471866752788,
        "valueF": 20.57149409499908,
        "lambda1": 1.2053651954011523e-248,
        "lambda2": 5.780011478917432e-9,
        "lambda3": 0.9999999942199885,
        "t": 37,
        "firm_id": 5,
        "demand": 0.7199689544830068
      },
      {
        "level_prices": 2.965517232422277,
        "log_dmd": -0.02484536033529161,
        "valueF": 20.571494111049947,
        "lambda1": 1.4505557115515455e-256,
        "lambda2": 3.3734280153561293e-9,
        "lambda3": 0.999999996626572,
        "t": 38,
        "firm_id": 5,
        "demand": 0.9754607452882172
      },
      {
        "level_prices": 2.965517238678143,
        "log_dmd": -0.38178597986219526,
        "valueF": 20.57149412676415,
        "lambda1": 4.3263247486779535e-266,
        "lambda2": 1.0173225033032047e-9,
        "lambda3": 0.9999999989826774,
        "t": 39,
        "firm_id": 5,
        "demand": 0.6826411365225564
      },
      {
        "level_prices": 2.9655172396093548,
        "log_dmd": -1.3627166162718765,
        "valueF": 20.571494129103275,
        "lambda1": 9.954884955957698e-274,
        "lambda2": 6.666064426658558e-10,
        "lambda3": 0.9999999993333935,
        "t": 40,
        "firm_id": 5,
        "demand": 0.25596447433227326
      },
      {
        "level_prices": 2.965517231594133,
        "log_dmd": -0.3427245021324657,
        "valueF": 20.57149410896972,
        "lambda1": 3.5208595688949157e-276,
        "lambda2": 3.685326381518402e-9,
        "lambda3": 0.9999999963146736,
        "t": 41,
        "firm_id": 5,
        "demand": 0.7098337423071296
      },
      {
        "level_prices": 2.9655172354895494,
        "log_dmd": 0.2615318941738275,
        "valueF": 20.571494118754675,
        "lambda1": 5.035323071435306e-284,
        "lambda2": 2.2182216164501064e-9,
        "lambda3": 0.9999999977817783,
        "t": 42,
        "firm_id": 5,
        "demand": 1.2989183688395105
      },
      {
        "level_prices": 2.9655172404263226,
        "log_dmd": -0.6819556071978559,
        "valueF": 20.571494131155426,
        "lambda1": 4.596053839755324e-295,
        "lambda2": 3.5891736615900694e-10,
        "lambda3": 0.9999999996410827,
        "t": 43,
        "firm_id": 5,
        "demand": 0.5056272166504613
      },
      {
        "level_prices": 2.965517240180036,
        "log_dmd": 0.5711992901720138,
        "valueF": 20.571494130536774,
        "lambda1": 4.0874895573520276e-301,
        "lambda2": 4.5167469577349026e-10,
        "lambda3": 0.9999999995483253,
        "t": 44,
        "firm_id": 5,
        "demand": 1.7703889888850495
      },
      {
        "level_prices": 2.9655172412803363,
        "log_dmd": -0.5877066468556666,
        "valueF": 20.57149413330064,
        "lambda1": 8.5988478254e-314,
        "lambda2": 3.727580745515856e-11,
        "lambda3": 0.9999999999627242,
        "t": 45,
        "firm_id": 5,
        "demand": 0.5556000118044343
      },
      {
        "level_prices": 2.9655172412778343,
        "log_dmd": -0.06723891518142153,
        "valueF": 20.571494133294355,
        "lambda1": 2.4273e-320,
        "lambda2": 3.8218038555670587e-11,
        "lambda3": 0.9999999999617819,
        "t": 46,
        "firm_id": 5,
        "demand": 0.9349717956941471
      },
      {
        "level_prices": 2.965517241345754,
        "log_dmd": 0.37342559745765835,
        "valueF": 20.571494133464963,
        "lambda1": 0,
        "lambda2": 1.2638153461295461e-11,
        "lambda3": 0.9999999999873619,
        "t": 47,
        "firm_id": 5,
        "demand": 1.452702474765111
      },
      {
        "level_prices": 2.965517241375053,
        "log_dmd": 0.10153851497375238,
        "valueF": 20.571494133538557,
        "lambda1": 0,
        "lambda2": 1.6033303829658727e-12,
        "lambda3": 0.9999999999983966,
        "t": 48,
        "firm_id": 5,
        "demand": 1.106872548738139
      },
      {
        "level_prices": 2.965517241378335,
        "log_dmd": -0.2766052631829959,
        "valueF": 20.571494133546803,
        "lambda1": 0,
        "lambda2": 3.6734865171089443e-13,
        "lambda3": 0.9999999999996326,
        "t": 49,
        "firm_id": 5,
        "demand": 0.7583537881965325
      },
      {
        "level_prices": 0.6896551724137931,
        "log_dmd": 1.5954076277711222,
        "valueF": 19.774116879259957,
        "lambda1": 0.4,
        "lambda2": 0.4,
        "lambda3": 0.2,
        "t": 0,
        "firm_id": 6,
        "demand": 4.930338405416354
      },
      {
        "level_prices": 2.2813681016943432,
        "log_dmd": -0.014973846976015389,
        "valueF": 17.558232825186312,
        "lambda1": 0.12172259945747912,
        "lambda2": 0.5936748728453238,
        "lambda3": 0.284602527697197,
        "t": 1,
        "firm_id": 6,
        "demand": 0.9851377035961483
      },
      {
        "level_prices": 1.7318851726730102,
        "log_dmd": 0.18620618229303026,
        "valueF": 17.10224136074976,
        "lambda1": 0.000004942027219218918,
        "lambda2": 0.6074817021386294,
        "lambda3": 0.3925133558341514,
        "t": 2,
        "firm_id": 6,
        "demand": 1.204670616523491
      },
      {
        "level_prices": 1.7197497066384715,
        "log_dmd": 0.43027599494863056,
        "valueF": 17.081029330740435,
        "lambda1": 1.1173171921963364e-7,
        "lambda2": 0.6120425417424555,
        "lambda3": 0.3879573465258252,
        "t": 3,
        "firm_id": 6,
        "demand": 1.5376818574138331
      },
      {
        "level_prices": 1.870997439576049,
        "log_dmd": 0.8664869739237384,
        "valueF": 17.350659174526577,
        "lambda1": 7.80862725283181e-10,
        "lambda2": 0.5550788879551615,
        "lambda3": 0.4449211112639758,
        "t": 4,
        "firm_id": 6,
        "demand": 2.378540285075427
      },
      {
        "level_prices": 2.4927927443204947,
        "log_dmd": -0.356214157314296,
        "valueF": 18.59073761542969,
        "lambda1": 3.308868101509502e-14,
        "lambda2": 0.3208962391519885,
        "lambda3": 0.6791037608479784,
        "t": 5,
        "firm_id": 6,
        "demand": 0.7003226249620307
      },
      {
        "level_prices": 2.4588583304006635,
        "log_dmd": 0.08712473259485248,
        "valueF": 18.518421686241183,
        "lambda1": 9.677169447901074e-19,
        "lambda2": 0.33367673270624354,
        "lambda3": 0.6663232672937563,
        "t": 6,
        "firm_id": 6,
        "demand": 1.0910327585784132
      },
      {
        "level_prices": 2.628417698337527,
        "log_dmd": -0.1804862591057223,
        "valueF": 19.305846073100394,
        "lambda1": 6.669116873701484e-25,
        "lambda2": 0.19838813958716514,
        "lambda3": 0.8016118604128348,
        "t": 7,
        "firm_id": 6,
        "demand": 0.834864152398182
      },
      {
        "level_prices": 2.775862068965517,
        "log_dmd": 0.19020110021867714,
        "valueF": 19.667661891594836,
        "lambda1": 3.9210613870096456e-31,
        "lambda2": 0.13906339572790266,
        "lambda3": 0.8609366042720973,
        "t": 8,
        "firm_id": 6,
        "demand": 1.2094928024692264
      },
      {
        "level_prices": 2.86076286741969,
        "log_dmd": -0.11132706037088907,
        "valueF": 20.308360073997996,
        "lambda1": 3.020396780865842e-40,
        "lambda2": 0.03945294603673993,
        "lambda3": 0.96054705396326,
        "t": 9,
        "firm_id": 6,
        "demand": 0.8946460977897871
      },
      {
        "level_prices": 2.9204978047747376,
        "log_dmd": 0.15731474584764632,
        "valueF": 20.458409145641845,
        "lambda1": 1.5850012017541374e-48,
        "lambda2": 0.016955372227695997,
        "lambda3": 0.9830446277723041,
        "t": 10,
        "firm_id": 6,
        "demand": 1.1703639231210117
      },
      {
        "level_prices": 2.955586393674079,
        "log_dmd": -0.8457517421934524,
        "valueF": 20.546548690483835,
        "lambda1": 1.1808074442907808e-58,
        "lambda2": 0.0037401893954766485,
        "lambda3": 0.9962598106045234,
        "t": 11,
        "firm_id": 6,
        "demand": 0.42923456316459363
      },
      {
        "level_prices": 2.947572000937697,
        "log_dmd": -1.0375564982208783,
        "valueF": 20.526417218967666,
        "lambda1": 8.871166406176229e-64,
        "lambda2": 0.0067585970494388045,
        "lambda3": 0.9932414029505612,
        "t": 12,
        "firm_id": 6,
        "demand": 0.35431940514906207
      },
      {
        "level_prices": 2.9165139013406556,
        "log_dmd": 0.4829554666522504,
        "valueF": 20.448401919749067,
        "lambda1": 7.558244270384313e-68,
        "lambda2": 0.01845580339118165,
        "lambda3": 0.9815441966088183,
        "t": 13,
        "firm_id": 6,
        "demand": 1.6208577211842423
      },
      {
        "level_prices": 2.9600829955436527,
        "log_dmd": -0.5202280441845731,
        "valueF": 20.55784377115114,
        "lambda1": 1.2243783245878735e-79,
        "lambda2": 0.0020466640160266966,
        "lambda3": 0.9979533359839733,
        "t": 14,
        "firm_id": 6,
        "demand": 0.5943849864742135
      },
      {
        "level_prices": 2.960675957620155,
        "log_dmd": -0.37669265953002595,
        "valueF": 20.559333241351307,
        "lambda1": 1.654896922364875e-86,
        "lambda2": 0.0018233406365649866,
        "lambda3": 0.998176659363435,
        "t": 15,
        "firm_id": 6,
        "demand": 0.6861269160630398
      },
      {
        "level_prices": 2.9623596582494294,
        "log_dmd": -0.20081188350647075,
        "valueF": 20.56356255383635,
        "lambda1": 3.87240983524673e-94,
        "lambda2": 0.0011892196203447092,
        "lambda3": 0.9988107803796553,
        "t": 16,
        "firm_id": 6,
        "demand": 0.8180663088454718
      },
      {
        "level_prices": 2.9641144873970493,
        "log_dmd": -0.3744547730551331,
        "valueF": 20.567970535097448,
        "lambda1": 1.0383297345976784e-102,
        "lambda2": 0.0005283099413709153,
        "lambda3": 0.9994716900586291,
        "t": 17,
        "firm_id": 6,
        "demand": 0.6876641095993299
      },
      {
        "level_prices": 2.964610920746513,
        "log_dmd": 0.4214500445238778,
        "valueF": 20.569217533362085,
        "lambda1": 2.235695273797194e-110,
        "lambda2": 0.00034134153702765124,
        "lambda3": 0.9996586584629723,
        "t": 18,
        "firm_id": 6,
        "demand": 1.524170068471922
      },
      {
        "level_prices": 2.9654134667762215,
        "log_dmd": -1.070416291321995,
        "valueF": 20.571233460590623,
        "lambda1": 2.963967007919587e-122,
        "lambda2": 0.0000390839414229079,
        "lambda3": 0.9999609160585771,
        "t": 19,
        "firm_id": 6,
        "demand": 0.3428657556668358
      },
      {
        "level_prices": 2.9652133576457556,
        "log_dmd": 0.604518856451504,
        "valueF": 20.570730803509473,
        "lambda1": 2.988942252259294e-126,
        "lambda2": 0.00011444971783217901,
        "lambda3": 0.9998855502821677,
        "t": 20,
        "firm_id": 6,
        "demand": 1.8303713256324396
      },
      {
        "level_prices": 2.965493899293368,
        "log_dmd": -1.1058497028881362,
        "valueF": 20.57143550021873,
        "lambda1": 4.216861533089596e-139,
        "lambda2": 0.00000879117522477341,
        "lambda3": 0.9999912088247753,
        "t": 21,
        "firm_id": 6,
        "demand": 0.3309295709311999
      },
      {
        "level_prices": 2.9654434150901854,
        "log_dmd": -0.19604083146699378,
        "valueF": 20.57130868820292,
        "lambda1": 6.539417726737775e-143,
        "lambda2": 0.000027804706293659358,
        "lambda3": 0.9999721952937064,
        "t": 22,
        "firm_id": 6,
        "demand": 0.8219786713979815
      },
      {
        "level_prices": 2.965484934853966,
        "log_dmd": 0.12860381319703795,
        "valueF": 20.571412982310957,
        "lambda1": 1.5698946499223872e-151,
        "lambda2": 0.00001216739266202778,
        "lambda3": 0.999987832607338,
        "t": 23,
        "firm_id": 6,
        "demand": 1.137239475629073
      },
      {
        "level_prices": 2.9655102620076357,
        "log_dmd": -1.1774337330730273,
        "valueF": 20.57147660196243,
        "lambda1": 7.233463441627435e-162,
        "lambda2": 0.0000026285945267395033,
        "lambda3": 0.9999973714054732,
        "t": 24,
        "firm_id": 6,
        "demand": 0.30806831055823986
      },
      {
        "level_prices": 2.965491449830273,
        "log_dmd": 0.12398689918843381,
        "valueF": 20.57142934737619,
        "lambda1": 2.681066311131462e-165,
        "lambda2": 0.000009713700286571829,
        "lambda3": 0.9999902862997133,
        "t": 25,
        "firm_id": 6,
        "demand": 1.1320010407697034
      },
      {
        "level_prices": 2.96551161333427,
        "log_dmd": 0.09468422483249261,
        "valueF": 20.571479996379768,
        "lambda1": 1.306601802918728e-175,
        "lambda2": 0.0000021196533268691906,
        "lambda3": 0.9999978803466731,
        "t": 26,
        "firm_id": 6,
        "demand": 1.0993116649868218
      },
      {
        "level_prices": 2.965515932531262,
        "log_dmd": 0.19630744616860063,
        "valueF": 20.5714908458345,
        "lambda1": 9.094037441114089e-186,
        "lambda2": 4.929427714466155e-7,
        "lambda3": 0.9999995070572285,
        "t": 27,
        "firm_id": 6,
        "demand": 1.216900979371229
      },
      {
        "level_prices": 2.965516997337172,
        "log_dmd": -0.8796567815650913,
        "valueF": 20.571493520536198,
        "lambda1": 1.836542339621404e-196,
        "lambda2": 9.191197408380816e-8,
        "lambda3": 0.999999908088026,
        "t": 28,
        "firm_id": 6,
        "demand": 0.41492529725663496
      },
      {
        "level_prices": 2.965516769353451,
        "log_dmd": -0.23234824938830864,
        "valueF": 20.571492947860524,
        "lambda1": 1.8131088500433955e-201,
        "lambda2": 1.7777597287352774e-7,
        "lambda3": 0.9999998222240272,
        "t": 29,
        "firm_id": 6,
        "demand": 0.7926700283820151
      },
      {
        "level_prices": 2.9655170178792156,
        "log_dmd": -0.2756946719154915,
        "valueF": 20.571493572136063,
        "lambda1": 6.763768016486848e-210,
        "lambda2": 8.417536018960902e-8,
        "lambda3": 0.9999999158246399,
        "t": 30,
        "firm_id": 6,
        "demand": 0.7590446530337496
      },
      {
        "level_prices": 2.9655171250960506,
        "log_dmd": -0.6134432056879116,
        "valueF": 20.571493841455613,
        "lambda1": 4.277076173701611e-218,
        "lambda2": 4.379499374034736e-8,
        "lambda3": 0.9999999562050061,
        "t": 31,
        "firm_id": 6,
        "demand": 0.5414832174710701
      },
      {
        "level_prices": 2.9655171152955755,
        "log_dmd": 0.04286552318161124,
        "valueF": 20.571493816837656,
        "lambda1": 1.6518030175570403e-224,
        "lambda2": 4.748608177035403e-8,
        "lambda3": 0.9999999525139182,
        "t": 32,
        "firm_id": 6,
        "demand": 1.0437975188427886
      },
      {
        "level_prices": 2.9655172085611916,
        "log_dmd": -0.747085985710245,
        "valueF": 20.571494051112936,
        "lambda1": 2.160333898616634e-234,
        "lambda2": 1.2360070618877682e-8,
        "lambda3": 0.9999999876399294,
        "t": 33,
        "firm_id": 6,
        "demand": 0.4737450431205848
      },
      {
        "level_prices": 2.96551719379748,
        "log_dmd": -0.7073035818732636,
        "valueF": 20.57149401402775,
        "lambda1": 4.245913481343629e-240,
        "lambda2": 1.7920429485289818e-8,
        "lambda3": 0.9999999820795704,
        "t": 34,
        "firm_id": 6,
        "demand": 0.4929716646859497
      },
      {
        "level_prices": 2.965517178108116,
        "log_dmd": -0.0034563060268809065,
        "valueF": 20.571493974617404,
        "lambda1": 5.141277433588145e-246,
        "lambda2": 2.3829410586437712e-8,
        "lambda3": 0.9999999761705893,
        "t": 35,
        "firm_id": 6,
        "demand": 0.996549660123202
      },
      {
        "level_prices": 2.9655172231656297,
        "log_dmd": -0.9231062323456545,
        "valueF": 20.571494087798037,
        "lambda1": 1.1818463063544148e-255,
        "lambda2": 6.859697750482373e-9,
        "lambda3": 0.9999999931403023,
        "t": 36,
        "firm_id": 6,
        "demand": 0.39728306895083965
      },
      {
        "level_prices": 2.965517202660354,
        "log_dmd": 0.11384895690186952,
        "valueF": 20.571494036290535,
        "lambda1": 1.9802618949089042e-260,
        "lambda2": 1.458246392602698e-8,
        "lambda3": 0.9999999854175361,
        "t": 37,
        "firm_id": 6,
        "demand": 1.12058285579476
      },
      {
        "level_prices": 2.965517232742456,
        "log_dmd": 0.7573931695356884,
        "valueF": 20.57149411185421,
        "lambda1": 1.0913345167933689e-270,
        "lambda2": 3.2528410246815392e-9,
        "lambda3": 0.999999996747159,
        "t": 38,
        "firm_id": 6,
        "demand": 2.132709355992603
      },
      {
        "level_prices": 2.965517240903808,
        "log_dmd": -0.7993740116795124,
        "valueF": 20.57149413235483,
        "lambda1": 2.3792318202931682e-284,
        "lambda2": 1.7908520793965968e-10,
        "lambda3": 0.9999999998209148,
        "t": 39,
        "firm_id": 6,
        "demand": 0.4496103268565222
      },
      {
        "level_prices": 2.965517240606896,
        "log_dmd": -0.8074846120599591,
        "valueF": 20.571494131609015,
        "lambda1": 8.838143795802663e-290,
        "lambda2": 2.9090927491774827e-10,
        "lambda3": 0.9999999997090908,
        "t": 40,
        "firm_id": 6,
        "demand": 0.4459784653662479
      },
      {
        "level_prices": 2.965517240102265,
        "log_dmd": 0.12109375944761419,
        "valueF": 20.571494130341424,
        "lambda1": 3.623851324631956e-295,
        "lambda2": 4.809649809178392e-10,
        "lambda3": 0.9999999995190351,
        "t": 41,
        "firm_id": 6,
        "demand": 1.1287307365766233
      },
      {
        "level_prices": 2.9655172410988975,
        "log_dmd": -0.7200393018286477,
        "valueF": 20.57149413284488,
        "lambda1": 1.828513872842022e-305,
        "lambda2": 1.05609942377582e-10,
        "lambda3": 0.9999999998943899,
        "t": 42,
        "firm_id": 6,
        "demand": 0.4867331260821361
      },
      {
        "level_prices": 2.9655172409959674,
        "log_dmd": -0.30918074729112016,
        "valueF": 20.571494132586327,
        "lambda1": 2.585462805014e-311,
        "lambda2": 1.4437591611570782e-10,
        "lambda3": 0.999999999855624,
        "t": 43,
        "firm_id": 6,
        "demand": 0.734048080832856
      },
      {
        "level_prices": 2.9655172411648016,
        "log_dmd": 0.031153907957162774,
        "valueF": 20.571494133010425,
        "lambda1": 2.4579e-319,
        "lambda2": 8.07888973604628e-11,
        "lambda3": 0.9999999999192111,
        "t": 44,
        "firm_id": 6,
        "demand": 1.0316442699306343
      },
      {
        "level_prices": 2.9655172413220363,
        "log_dmd": 0.2842555178643825,
        "valueF": 20.571494133405384,
        "lambda1": 0,
        "lambda2": 2.1570703951249033e-11,
        "lambda3": 0.9999999999784293,
        "t": 45,
        "firm_id": 6,
        "demand": 1.3287724124682105
      },
      {
        "level_prices": 2.96551724137049,
        "log_dmd": -0.29770032254032647,
        "valueF": 20.571494133527096,
        "lambda1": 0,
        "lambda2": 3.3219889713113216e-12,
        "lambda3": 0.9999999999966781,
        "t": 46,
        "firm_id": 6,
        "demand": 0.7425238240626844
      },
      {
        "level_prices": 2.9655172413744957,
        "log_dmd": 0.6535776159245041,
        "valueF": 20.571494133537158,
        "lambda1": 0,
        "lambda2": 1.8130729827104736e-12,
        "lambda3": 0.9999999999981869,
        "t": 47,
        "firm_id": 6,
        "demand": 1.9224061718450833
      },
      {
        "level_prices": 2.965517241378978,
        "log_dmd": -0.7755827120703818,
        "valueF": 20.57149413354842,
        "lambda1": 0,
        "lambda2": 1.2509354336857596e-13,
        "lambda3": 0.9999999999998748,
        "t": 48,
        "firm_id": 6,
        "demand": 0.4604354015464745
      },
      {
        "level_prices": 2.9655172413787976,
        "log_dmd": -0.3861814183462729,
        "valueF": 20.57149413354796,
        "lambda1": 0,
        "lambda2": 1.9296075065460392e-13,
        "lambda3": 0.9999999999998069,
        "t": 49,
        "firm_id": 6,
        "demand": 0.6796472140215439
      }
    ]
  }
}
    vegaEmbed("#vis", yourVlSpec);
  </script>
  
 {% endraw %}




The true lambda that generated the demand data was lambda3. Because of this, a firm that learns correctly is one that puts probability one to lambda3.

The graph shows that some firms learnt this quite fast (firm 0, 5, 6), while others took longer or haven't converged to the right result after 50 periods (for example, firm 2 and 3).

If you select firm number 3, you can see why this might have happened: the demand is random and so big errors (the $$\varepsilon$$ in our demand equation) might make the demand appear **as if** the correct value is, say lambda2. If the firm is unlucky, it might get stuck in the wrong lambda for quite a while. However, we know that active learning is the correct approach and we're assured that the learning will eventually converge. This is not true if the firm were to use passive learning: in this case, there is a non trivial probability that the firm gets stuck forever in the wrong lambda. 


### Some notes on how to write this Altair interactive plot

If you're interested in Altair, I suggest you take a look at Jake Vanderplas' amazing tutorial [here](https://www.youtube.com/watch?v=ms29ZPUKxbU). In the meantime, I can give you some intuition on how the syntax works.


This graph has three components: the base graph (and the timeseries plots that come from it), a selector object and a legend.

Let's take a look at the base graph. The four time series graphs take this as the base.

```
base = alt.Chart(all_firms).properties(
    width=250,
    height=250
).add_selection(selector).transform_filter(
    selector
)
```

This tells `altair` to use the `all_firms` dataframe for the chart, specifies width, height and then adds both a selection mechanism and a `transform_filter` method. What does this do? It's telling `altair` to use the `selector`object to do some filtering. Thus, if a firm_id is selected in some way, only the date for that firm_id will be passed onto the graphs. I'll explain a bit more how this works below, but let's first take a look at one of four timeseries plots:


```
timeseries1 = base.mark_line(strokeWidth=2).encode(
    x=x_for_tseries,
    y=alt.Y('level_prices'),
    color=color_timeseries
)
```

We take the `base` graph we defined before and then use `mark_line`, which means we'll represent parts of the data as a line (instead of representing them as points, areas or whatnot). With the encoding, we tell `altair` to use a certain `x` and `y` axis and a color.


Now let's go back to the second component: the selector object.

```
selector = alt.selection_single(empty='all', fields=['firm_id'], on='mouseover')
```

A `selection_single` means we can only select one "thing". What kind of thing? One single `firm_id`. We will allow the selection to happen upon mouseover (as opposed, to, say, a click).


Finally, we have the legend. One of the cool things about `altair` is that you can create new objects and functionalities if you know how to combine the building blocks

```
legend = alt.Chart(all_firms).mark_point(size=400).encode(
    y=alt.Y('firm_id:N', axis=alt.Axis(orient='right')),
    color=color
).add_selection(
    selector
)
```

If you pay attention, you'll see there's no `legend` function invoked. What I'm doing here is to encode the `firm_id` as points with some color. It's not a specially crafted `legend`function, it's just a set of points that can be used as a clickable legend now that I add the same `selector object`. Cool, right??

To finish our plot, we put all these elements together

```
((timeseries1 | timeseries2) & (timeseries3 | timeseries4) | legend).properties(
title='Learning for multiple firms. Hover on firm-id')
```

The `|` is used to put charts side by side and `&` to put one on top of the other.


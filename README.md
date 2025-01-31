# NDP
The dataset and code for the accepted ICSE25 paper "Faster Configuration Performance Bug Testing with Neural Dual-level Prioritization"

## Abstract
> As software systems become more complex and
> configurable, more performance problems tend to arise from
> the configuration designs. This has caused some configuration
> options to unexpectedly degrade performance which deviates
> from their original expectations designed by the developers. Such
> discrepancies, namely configuration performance bugs (CPBugs),
> are devastating and can be deeply hidden in the source code.
> Yet, efficiently testing CPBugs is difficult, not only due to the
> test oracle is hard to set, but also because the configuration
> measurement is expensive and there are simply too many possible
> configurations to test. As such, existing testing tools suffer from
> lengthy runtime or have been ineffective in detecting CPBugs
> when the budget is limited, compounded by inaccurate test oracle.
> In this paper, we seek to achieve significantly faster CP-
> Bug testing by neurally prioritizing the testing at both the
> configuration option and value range levels with automated
> oracle estimation. Our proposed tool, dubbed **NDP**, is a general
> framework that works with different heuristic generators. The
> idea is to leverage two neural language models: one to estimate
> the CPBug types that serve as the oracle while, more vitally, the
> other to infer the probabilities of an option being CPBug-related,
> based on which the options and the value ranges to be searched
> can be prioritized. Experiments on several widely-used systems of
> different versions reveal that **NDP** can, in general, better predict
> CPBug type in **87%** cases and find more CPBugs with up to
> **88.88×** testing efficiency speedup over the state-of-the-art tools


This repository contains the key codes, full data used and raw experiment results for the paper.


## Folders

* [RQ1](https://github.com/ideas-labo/ndp/tree/main/rq1): Contains data, models and key codes used to classify CPBugs types. See the rq1 folder for details.


* [RQ2](https://github.com/ideas-labo/ndp/tree/main/rq2): Contains data, models and key codes used to Option-CPBugs Relevance Estimation. See the rq2 folder for details.


* [RQ3](https://github.com/ideas-labo/ndp/tree/main/rq3): Contains data and key codes used to CPBugs testing framework. See the rq3 folder for details.


* [NDP](https://github.com/ideas-labo/ndp/tree/main/ndp): Contains deploying *NDP* in the real-world system. See the ndp folder for details.

# antagonizer

Based on a corpus of _authors_, having produced a number of _documents_ each, this method will do the following:

1. Tag a number of the authors as belonging to either of two user-provided groups. The group affiliation is based on user-provided lists of indentifying words.
2. Calculate a bias_score for phrases, reflecting partisanship of language-use. The method used draws on a workflow described [here](https://towardsdatascience.com/detecting-politically-biased-phrases-from-u-s-senators-with-natural-language-processing-tutorial-d6273211d331). The maths for calculating bias is based on the paper _Auditing the partisanship of Google search snippets_ ([Hu et al. , 2019](https://dl.acm.org/doi/10.1145/3308558.3313654)). Phrases with a bias score of -1.0 are used only by the first group, and phrases with a bias score of 1.0 are used only by the second group.
3. 

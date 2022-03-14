# antagonizer

In their understanding of the social as a discursive space, Laclau and Mouffe (1985, p. 114) argue that the importance of the concept of _**antagonism**_ "stems from the fact that, in one of its key dimensions, the specificity of a hegemonic articulatory practice is given by its confrontation with other articulatory practices of an antagonistic character". The hereby provided method makes it possible to map antagonistic relationships in (political) discourse. Laclau and Mouffe (1985, p. 124) go on to say that "the usual descriptions of antagonisms in the sociological or historical literature [...] explain the conditions which made antagonisms possible, but not the antagonisms as such". 

This method makes it possible to get a visual image of the polarity/partisanship of (political) speech, and thereby enables further closer analysis of the shape of antagonisms, in terms of the intensity and tonality of the different sides of a debate or issue.

![example-image](https://github.com/simonlindgren/antagonizer/blob/main/antagonizer.png)

Based on a corpus of _authors_, having produced a number of _documents_ each, this method will do the following:

1. Tag a number of the authors as promoting either of two partisan discourses, or a hybrid form of the two. The discursive orientation is based on user-provided lists of indentifying words.
2. Calculate a _bias score_ for phrases, reflecting partisanship of language-use. The method used for this step draws on a workflow described [here](https://towardsdatascience.com/detecting-politically-biased-phrases-from-u-s-senators-with-natural-language-processing-tutorial-d6273211d331). The maths for calculating bias is based on the paper _Auditing the partisanship of Google search snippets_ ([Hu et al. , 2019](https://dl.acm.org/doi/10.1145/3308558.3313654)). Phrases with a bias score of -1.0 are used only by the first group, and phrases with a bias score of 1.0 are used only by the second group.
3. Assign the tagged authors with individual scores (mean bias score of their used partisan phrases) based on their language use.
4. Draw plots showing the polarization (antagonism) between the two groups.

The code is in `antagonizer.py`. See the tutorial notebook for details on how to run the analysis.

----
Laclau, E., & Mouffe, C. (1985). _Hegemony and Socialist Strategy: Towards a Radical Democratic Politics_. Verso.

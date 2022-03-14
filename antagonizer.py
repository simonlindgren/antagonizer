#!/usr/bin/env python3

'''
antagonizer.py
----
by Simon Lindgren
github.com/simonlindgren/antagonizer
'''

# Import packages
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore') # use this to suppress the numpy warning during score_authors
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.tokenize import wordpunct_tokenize
stop_words = [i.strip() for i in open('stopwords.txt', 'r')]

# Prepare data
def prepare_data(df, threshold):
   # Read data and filter by threshold
    print('read ...')
    counts_df = pd.DataFrame(df.author.value_counts()).reset_index()
    counts_df.columns = ['author', 'numdocs']
    counts_df = counts_df[counts_df['numdocs'] > threshold]

    # Create df of docs-per-author
    print('merge ...')
    df = pd.merge(df, counts_df, on = 'author')
    df = pd.DataFrame(df.groupby('author'))
    df.columns = ['author', 'data']
    docs = []
    for d in df.data:
        doc = []
        for i in d.text:
            doc.append(str(i))
        docs.append(' '.join(doc))
    df['doc'] = docs
    df = df.drop('data', axis=1)

    # Preprocess the docs
    print('preprocess ...')
    docs = []
    for d in df.doc:
        d = wordpunct_tokenize(d)
        d = [w.lower() for w in d if w.isalpha() and w not in stop_words \
             and len(w) > 3 and 'http' not in w]
        docs.append(' '.join(d))
    df['doc'] = docs

    # Add bigrams
    print('bigrams ...')
    full_docs = []
    for doc in df.doc:
        d = doc.split()
        bigrams = nltk.bigrams(d)
        bgms = []
        for b in bigrams:
            bgms.append(b[0] + ' ' + b[1])
        d = d + bgms
        full_docs.append(d)

    df['full_doc'] = full_docs

    df = pd.merge(df,counts_df, on = 'author')
    df.to_csv('prep_df.csv', index = False)
    print('done!')
    return df

# Categorize
def categorize(prep_df,cat1,cat2,cat3,tags1,tags2):
    print('categorizing authors ...')
    labels = []
    
    # This is the labelling logic:
    # See how many cat1 tags and how many cat2 tags each doc matches
    for d in prep_df.doc:
        # Set label as empty to begin with
        label = ''
        
        # Count the author's number of matches for both cat1 and cat2
        cat1_matches = 0
        for t in tags1:
            cat1_matches = cat1_matches + d.count(t)
        cat2_matches = 0
        for t in tags2:
            cat2_matches = cat2_matches + d.count(t)
        
        # If there are any matches at all
        if cat1_matches + cat2_matches > 0:
            # Calculate the share of cat1 matches and cat2 matches
            cat1_share = cat1_matches/(cat1_matches+cat2_matches)
            cat2_share = 1 - cat1_share
            # If 80% dominance or more for cat1 matches
            if cat1_share > 0.8:
                # Tag author as cat1
                label = cat1
            # If 80% dominance or more for cat2 matches
            if cat2_share > 0.8:
                # Tag author as cat2
                label = cat2
            else:
                # If author has BOTH cat1 matches and cat2 matches...
                # ... with none of them having over 80% dominance
                if cat1_matches > 0 < 0.8 and cat2_matches > 0 < 0.8:
                    # Tag author as cat3
                    label = cat3
        # Set the label
        # If none of the criteria above are met, label will remain empty
        labels.append(label)
    prep_df['category'] = labels
    cat_df = prep_df[prep_df['category'] != '']
    for cat,count in zip(pd.DataFrame(cat_df.category.value_counts()).index,pd.DataFrame(cat_df.category.value_counts()).category):
        print(cat,count)
    cat_df.to_csv('cat_df.csv', index = False)
    print('done!')
    return cat_df

# Partisan phrases
def partisan_phrases(cat_df,max_df, min_df, cat1, cat2):
    print('vectorize ...')
    tf_vectorizer = CountVectorizer(max_df=max_df, min_df=min_df,
    ngram_range = (1,2),
    stop_words=stop_words)

    term_frequencies = tf_vectorizer.fit_transform(cat_df.doc.tolist())

    phrases_df = pd.DataFrame(data=tf_vectorizer.get_feature_names_out(),columns=['phrase'])

    print('analyze partisan patterns ...')
    cat1_tfs = tf_vectorizer.transform(cat_df[cat_df.category==cat1].doc.tolist())
    cat2_tfs = tf_vectorizer.transform(cat_df[cat_df.category==cat2].doc.tolist())
    n_cat1_docs = cat1_tfs.shape[0]
    n_cat2_docs = cat2_tfs.shape[0]

    total_cat1_tfs = cat1_tfs.sum(axis=0)
    total_cat2_tfs = cat2_tfs.sum(axis=0)
    total_tfs = total_cat1_tfs + total_cat2_tfs
    p_cat1 = total_cat1_tfs / n_cat1_docs
    p_cat2 = total_cat2_tfs / n_cat2_docs

    try:
        bias = (p_cat2 - p_cat1) / (p_cat2 + p_cat1)
    except:
        bias = 0

    phrases_df['bias_score'] = bias.T
    phrases_df['p_' + cat1] = p_cat1.T
    phrases_df['p_' + cat2] = p_cat2.T
    phrases_df['n_' + cat1] = total_cat1_tfs.T
    phrases_df['n_' + cat2] = total_cat2_tfs.T
    phrases_df = phrases_df.dropna()
    phrases_df = phrases_df.sort_values(by='bias_score')
    phrases_df.to_csv('phrases_df.csv', index = False)
    print('done!')
    return phrases_df

#Score authors
def score_authors(phrases_df, cat_df):

    bias_score_dict = dict(zip(phrases_df.phrase, phrases_df.bias_score))


    print('scoring authors ...')
    authorscores = []
    for d in cat_df.full_doc:
        scores =  []
        for w in d:
            if bias_score_dict.get(w):
                scores.append(bias_score_dict.get(w))
        authorscores.append(np.mean(scores))

    cat_df['authorscores'] = authorscores

    final_df = cat_df
    final_df = final_df.dropna()
    final_df.sort_values(by='authorscores')
    final_df.to_csv('final_df.csv', index = False)
    print('done!')
    return final_df

# Reduce dataframe, remove less active authors, to make plotting less demanding
def reduce(final_df, cat1,cat2,cat3):
    print('reduce ...')
    cat1_data = final_df[final_df.category == cat1].sort_values(by='numdocs', ascending = False)[:50000]
    cat2_data = final_df[final_df.category == cat2].sort_values(by='numdocs', ascending = False)[:50000]
    cat3_data = final_df[final_df.category == cat3].sort_values(by='numdocs', ascending = False)[:50000]

    potential_cutoffs = []
    potential_cutoffs.append(list(cat1_data.numdocs)[-1])
    potential_cutoffs.append(list(cat2_data.numdocs)[-1])
    potential_cutoffs.append(list(cat3_data.numdocs)[-1])

    actual_cutoff = int(min(potential_cutoffs))
    cat1_data = final_df[final_df.category == cat1]
    cat2_data = final_df[final_df.category == cat2]
    cat3_data = final_df[final_df.category == cat3]
    cat1_data = cat1_data[cat1_data.numdocs >= actual_cutoff]
    cat2_data = cat2_data[cat2_data.numdocs >= actual_cutoff]
    cat3_data = cat3_data[cat3_data.numdocs >= actual_cutoff]
    plot_df = pd.concat([cat1_data,cat2_data,cat3_data])
    plot_df = plot_df.sort_values(by='numdocs',ascending=False)
    plot_df.to_csv('plot_df.csv', index = False)
    print(str(len(plot_df)) + ' authors in plot_df')
    print('done!')
    return plot_df

# Plot
def plot(plot_df,plotcut,colour1,colour2,colour3, width,height):
    from bokeh.io import output_notebook
    output_notebook()
    from bokeh.plotting import figure, show
    from bokeh.plotting import figure, show
    from bokeh.sampledata.penguins import data
    from bokeh.transform import factor_cmap, factor_mark
    from bokeh.models import HoverTool
    print('preparing plot ...')
    SPECIES = sorted(plot_df.category.unique())
    MARKERS = ['hex', 'triangle', 'circle_x']
    TOOLTIPS = [
        ("author", "@author"),
        ("authorscore", "@authorscores"),
        ("numdocs", "@numdocs"),
        ('category', '@category')
    ]

    p = figure(title = ' ',
               y_axis_type="log",
               width = width,
               height = height,
               background_fill_color="#fafafa",
               tooltips = TOOLTIPS
              )
    p.xaxis.axis_label = 'Polarity'
    p.yaxis.axis_label = 'Number of documents'

    p.scatter("authorscores", "numdocs", source=plot_df[:plotcut], 
              legend_group="category", fill_alpha=0.4, size=12,
              marker=factor_mark('category', MARKERS, SPECIES),
              color=factor_cmap('category', palette=[colour2,colour1,colour3], factors = SPECIES))

    p.legend.location = "top_right"
    p.legend.background_fill_alpha = 0.2
    p.legend.border_line_alpha = 0.0
    
    
    p.legend.title = "Discursive orientation"

    show(p)
    return p

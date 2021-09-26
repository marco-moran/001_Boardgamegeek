#!/usr/bin/env python
# coding: utf-8

# # Migliore di Tutti

# ## Import Library

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from operator import itemgetter
from typing import List
import scipy.stats
import seaborn as sn
plt.style.use('ggplot')


# ## Class and Method

# In[2]:


class ItemRank(object):
    """
    This class ranks Pandas dataframes using a specific field and implementing different ranking methodologies
    """
    def __init__(self, 
                 dataframe=pd.DataFrame,
                 df_key=List[str], 
                 rating=None,
                 m=None, 
                 C=None,  **args):
        self.data = dataframe
        self.df_key = df_key
        self.rating = rating
        self.prior = m
        self.confidence = C
        
    @property
    def items(self):
        ##Returns the data grouped by items
        return self.data.groupby(self.df_key)

    def get_means(self):
        return self.items[self.rating].mean()

    def get_counts(self):
        return self.items[self.rating].count()
    
    def plot_mean_frequency(self):
        grid   = pd.DataFrame({
                    'Mean Rating':  self.items[self.rating].mean(),
                    'Number of Reviews': self.items[self.rating].count()
                 })
        grid.plot(x='Number of Reviews', y='Mean Rating', kind='hexbin',
                  xscale='log', cmap='YlGnBu', gridsize=12, mincnt=1,
                  title="Ratings by Simple Mean")
        plt.show()
    
    def bayesian_mean(self, arr):
        if not self.prior or not self.confidence:
            raise TypeError("Bayesian mean must be computed with m and C")

        return ((self.confidence * self.prior + arr.sum()) /
                (self.confidence + arr.count()))
    
    def get_bayesian_estimates(self):
        return self.items[self.rating].agg(self.bayesian_mean)
    
    def top_items(self, n=10):
        table   = pd.DataFrame({
                    'count': self.get_counts(),
                    'mean':  self.get_means(),
                    'bayes': self.get_bayesian_estimates()
                    
                 })
        return table.sort_values('mean', ascending = False)[:n]

    def get_rank(self,rating_method='avg',ascending = True):
        if rating_method == 'bayes':
            table   = pd.DataFrame({
                    'count': self.get_counts(),
                    'rating': self.get_bayesian_estimates()
                 })
        elif rating_method == 'avg':
            table   = pd.DataFrame({
                    'count': self.get_counts(),
                    'rating': self.get_means()
                 }).reset_index(level=self.df_key)
        table1 = table.sort_values(['rating', 'count', 'game'], ascending=False).reset_index() 
        table1['rank'] = table1.index + 1
        return table1.sort_values('rank')


# In[36]:


def rank_comparison(df,rating_1,rating_2,count_1,count_2,rank_1,rank_2,x_label, n1=30, n2=10):
    
    fig, ax = plt.subplots(figsize=(16, 18), nrows=2, ncols=3)
    fig.suptitle("Rank Comparison", fontsize=16)
    
    # heat map for the correlation of Spearman, Pearson and Kendall
    r, s, k = (df[[rating_1,rating_2]].corr(), df[[rating_1,rating_2]].corr(method='spearman'), df[[rating_1,rating_2]].corr(method='kendall'))
    sn.set(font_scale=1.0)
    sn.heatmap(r, vmin=-1, vmax=1, annot=True, annot_kws={"size": 16}, ax=ax[0,0]).set_title("Pearson-on mean", fontweight='bold')
    sn.heatmap(s, vmin=-1, vmax=1, annot=True, annot_kws={"size": 16}, ax=ax[0,1]).set_title("Spearman-on rank", fontweight='bold')
    sn.heatmap(k, vmin=-1, vmax=1, annot=True, annot_kws={"size": 16}, ax=ax[0,2]).set_title("Kendall-on rank", fontweight='bold')    
           
    # bar chart of the top n1 games of rank_1 with their number of reviews
    ax[1,0].bar(df.sort_values(rank_1)[rank_1][:n1], df.sort_values(rank_1)[count_1][:n1], color='#7f6d5f')
    ax[1,0].set_title(rank_1, fontweight='bold')
    ax[1,0].set_xlabel('Rank', fontweight='bold')
    ax[1,0].set_ylabel('Reviews', fontweight='bold')
    ax[1,0].set_ylim(0, df[count_1].max())
    
    # bar chart of the top n1 games of rank_2 with their number of reviews
    ax[1,1].bar(df.sort_values(rank_2)[rank_2][:n1], df.sort_values(rank_2)[count_2][:n1], color='#557f2d')
    ax[1,1].set_title(rank_2, fontweight='bold')
    ax[1,1].set_xlabel('Rank', fontweight='bold')
    ax[1,1].set_ylabel('Reviews', fontweight='bold')
    ax[1,1].set_ylim(0, df[count_2].max())
    
    # bar chart comparing the ratings of the top np2 games in the rank_1 with their rating according to the rank_2
    t = df.sort_values(rank_1)
    i = t[rank_1][0:n2].index.tolist()
    x = df[x_label].iloc[i].tolist()
    y = df[rating_1].iloc[i].tolist()
    y1 = df[rating_2].iloc[i].tolist()
    
    barWidth = 0.25
    r1 = np.arange(len(y))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    plt.bar(r1, y, color='#7f6d5f', width=barWidth, edgecolor='white', label=rank_1)
    plt.bar(r2, y1, color='#557f2d', width=barWidth, edgecolor='white', label=rank_2)
    
    plt.title('top ' + str(n2) + ' by rank', fontweight='bold')
    plt.xlabel('game', fontweight='bold')
    plt.ylabel('rating', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(y))], x, rotation=45, horizontalalignment='right')
    plt.legend()   



    plt.show()


# ## Data Exploration

# In[4]:


# file path
bgg = 'data/bgg.csv'


# In[5]:


#import data
df = pd.read_csv(bgg)


# In[6]:


df.dtypes


# In[7]:


df.info()


# In[8]:


df.head()


# In[9]:


df['rating'].describe().apply("{0:.5f}".format)


# In[10]:


df.isna().any()


# In[11]:


df['rating'].nunique()


# In[12]:


fig, ax = plt.subplots(figsize=(16, 6),nrows=1, ncols=2)
df[['rating']].boxplot(ax=ax[0])
df['rating'].plot.kde(ax=ax[1])
plt.show()


# ## ranking with bayes

# In[13]:


ratings = ItemRank(df,df_key= ['game','title'], rating = 'rating',m=5,C=30)


# In[14]:


print (ratings.top_items(n=10))
print (ratings.plot_mean_frequency())


# In[15]:


bayes_rank = ratings.get_rank(rating_method='bayes',ascending= False)


# In[16]:


bayes_rank[:10]


# ## Complete BBG Dataset

# In[17]:


bgg_true = pd.read_csv('data/bgg_true_stats.csv')
bgg_true.head()


# In[18]:


bgg_true_filtered= bgg_true[bgg_true['rank']!='Not Ranked']


# In[19]:


bgg_true_filtered


# In[20]:


bgg_true_filtered['rating'] = pd.to_numeric(bgg_true_filtered['rating'])
bgg_true_filtered['rank'] = pd.to_numeric(bgg_true_filtered['rank'])


# In[21]:


bgg_true_filtered.count()


# ## Bayes vs Bgg

# In[33]:


full_df=pd.merge(bayes_rank, bgg_true_filtered, how="inner", on=["game","game"],suffixes=('_bayes', '_bgg'))


# In[34]:


full_df.head()


# In[37]:


rank_comparison(df=full_df,rating_1='rating_bayes',rating_2='rating_bgg',count_1='count_bayes',count_2='count_bgg',
                rank_1='rank_bayes',rank_2='rank_bgg',x_label= 'title_bayes')


# ## Avg vs Bgg

# In[25]:


avg_rank = ratings.get_rank(rating_method='avg',ascending= False)


# In[26]:


full_df_avg=pd.merge(avg_rank, bgg_true_filtered, how="inner", on=["game","game"],suffixes=('_avg', '_bgg'))


# In[27]:


full_df_avg.head()


# In[38]:


rank_comparison(df=full_df_avg,rating_1='rating_avg',rating_2='rating_bgg',count_1='count_avg',count_2='count_bgg',
                rank_1='rank_avg',rank_2='rank_bgg',x_label= 'title_avg')


# ## Avg vs Bayes

# In[29]:


full_Avg_B = pd.merge(avg_rank, bayes_rank, how="inner", on=["game","game"],suffixes=('_avg', '_bayes'))


# In[30]:


full_Avg_B.head()


# In[39]:


rank_comparison(df=full_Avg_B,rating_1='rating_avg',rating_2='rating_bayes',count_1='count_avg',count_2='count_bayes',
                rank_1='rank_avg',rank_2='rank_bayes',x_label= 'title_bayes')


# In[ ]:





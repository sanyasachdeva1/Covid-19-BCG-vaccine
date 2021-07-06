#!/usr/bin/env python
# coding: utf-8

# In[234]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import openpyxl


# In[235]:


df = pd.read_csv("task_2-Gemany_per_state_stats_20June2020.csv")
df


# In[236]:


df.describe()
df.dtypes


# In[237]:


df['Cases']


# In[239]:


df.isnull().sum()


# In[240]:


df = df.drop("State in Germany (German)", axis=1)


# In[241]:


import pandas_profiling as pp
pp.ProfileReport(df)


# In[242]:


import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[243]:


pip install plotly


# In[244]:


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px
import numpy as np


# In[245]:


dist = df['East/West'].value_counts()
colors = ['mediumturquoise', 'darkorange']
trace = go.Pie(values=(np.array(dist)),labels=dist.index)
layout = go.Layout(title='Location- East/West')
data = [trace]
fig = go.Figure(trace,layout)
fig.update_traces(marker=dict(colors=colors, line=dict(color='#000000', width=2)))
fig.show()


# Most of the states are in West 

# In[246]:


def df_to_plotly(df):
    return {'z': df.values.tolist(),
            'x': df.columns.tolist(),
            'y': df.index.tolist() }
import plotly.graph_objects as go
dfNew = df.corr()
fig = go.Figure(data=go.Heatmap(df_to_plotly(dfNew)))
fig.show()


# In[247]:


#s = [df['Population'], df['East/West']]
fig = px.scatter(df, x='Population', y='Deaths')
fig.update_traces(marker_color="turquoise",marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5)
fig.update_layout(title_text='Population and Deaths')
fig.show()


# As the population increases the number of cases increasing but  the number of states decreases 

# In[248]:


#s = [df['Population'], df['East/West']]
fig = px.scatter(df, x='Deaths', y='East/West')
fig.update_traces(marker_color="turquoise",marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5)
fig.update_layout(title_text='Deaths and East/West')
fig.show()


# Most of the deaths were in West 

# In[256]:


import plotly.express as px
fig = px.histogram(df, x="Deaths", y="Cases", color='State in Germany (English)' ,title='Cases in each state')
fig.show()


# In[263]:


import plotly.express as px
fig = px.scatter(df, x="Cases", y="Population", color='State in Germany (English)' ,title='Cases in each state')
fig.show()



# coding: utf-8

# # Random Variables

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


n = np.random.randint(2, 10, 40)
print(n)


# ## Probability Mass Function

# In[ ]:


#Convert list n to dataframe
df = pd.DataFrame(n)

#Count each variable how many times repeated
df = pd.DataFrame(df[0].value_counts())
df


# In[ ]:


length = len(n)
length


# In[ ]:


df.columns = ["Counts"]
df


# In[ ]:


df["Prob"] = df["Counts"] / length
df


# In[ ]:


# Plot pmf
plt.bar(df["Counts"], df["Prob"])


# In[ ]:


sns.barplot(df["Counts"], df["Prob"])


# In[ ]:


data = {"Candy" : ["Blue", "Orange", "Green", "Purple"],
        "Total" : [30000, 10000, 20000, 12000]}

df = pd.DataFrame(data)
df


# In[ ]:


df["pmf"] = df["Total"]/df["Total"].sum()
df


# In[ ]:


plt.bar(df["Candy"], df["pmf"])


# In[ ]:


#Plot pmf using seaborne
sns.barplot(df["Candy"], df["pmf"])


# ## Probability Density Function

# In[ ]:


data = np.random.normal(size = 100)
data = np.append(data, [1.2, 1.2, 1.2, 1.2, 1.2])
sns.distplot(data)


# In[ ]:


dir(sns)


# In[ ]:


import scipy.stats as stats
mu = 20
sigma = 2

h = sorted(np.random.normal(mu, sigma, 100))


# In[ ]:


from scipy.interpolate import fitpack2
plt.figure(figsize = (10, 5))

fit = stats.norm.pdf(h, np.mean(h), np.std(h))

plt.plot(h, fit, "-o", color = "Black")

plt.hist(h, density = True)


# ## Cumulative Distribution Function

# In[ ]:


import scipy.stats as ss

x  = np.linspace(-5, 5, 5000)
mu = 0
sigma = 1

y_pdf = ss.norm.pdf(x, mu, sigma)
y_cdf = ss.norm.cdf(x, mu, sigma)

plt.plot(x, y_pdf, label = "pdf")
plt.plot(x, y_cdf, label = "cdf")


# In[ ]:


plt.figure(figsize = (10, 5))

fit = stats.norm.cdf(h, np.mean(h), np.std(h))

plt.plot(h, fit, "-o", color = "Black")

plt.hist(h, density = True)


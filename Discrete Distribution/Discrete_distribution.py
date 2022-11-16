
# coding: utf-8

# In[ ]:


from numpy import random as r
import numpy as np
from IPython.display  import Math,Latex
from IPython.core.display import Image
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


sns.set(color_codes = True)
sns.set(rc = {"figure.figsize":(5,5)})


# ## Uniform distribution

# In[ ]:


from scipy.stats import randint

fig , ax = plt.subplots(1,1)


low, high = 7,31

mean , var , skew, kurt  = randint.stats(low,high,moments= "mvsk")

x = np.arange(randint.ppf(0.01,low,high),
                randint.ppf(0.99,low,high))
print(x)

ax.plot(x, randint.pmf(x,low,high), "bo",ms = 8,label="randint pmf")
ax.vlines(x,0, randint.pmf(x,low,high),color= "b",lw = 5,alpha = 0.5)

rv = randint(low,high)
ax.vlines(x,0,rv.pmf(x),color = "k",linestyle="-",lw = 1,label="frozen pmf")
ax.legend(loc = "best",frameon = False)
plt.show()


# In[ ]:


x


# In[ ]:


randint.ppf(0.01,low,high)


# In[ ]:


randint.ppf(0.99,low,high)


# In[ ]:


uniformMatrix = r.uniform(0.2, 0.4, size=(10))
print(uniformMatrix)


# In[ ]:


sns.distplot(r.uniform(size=1000), hist=False)


# ## Bernouli Distribution

# In[ ]:


from scipy.stats import bernoulli
#generating random variables (rvs)
data_bern = bernoulli.rvs(size=10000, p=0.6)


# In[ ]:


ax = sns.distplot(data_bern,
                  kde=False,
                  color="skyblue",
                  hist_kws={"linewidth":15, "alpha":1})
ax.set(xlabel="Bernoulli Distribution", ylabel="Frequency")


# ## Binomial Distribution
# P(x) = (n!/(n-x)x!)p^x * q^(n-x)

# In[ ]:


from scipy.stats import binom
data_binom = binom.rvs(n=10, p=0.8, size=10000)


# In[ ]:


ax = sns.distplot(data_binom,
                  kde=False,
                  color="skyblue",
                  hist_kws={"linewidth":15, "alpha":1})

ax.set(xlabel="Binomial Distribution", ylabel="Frequency")


# ## Poisson Distribution

# In[ ]:


from scipy.stats import poisson
data_poisson = poisson.rvs(mu=3, size = 10000)


# In[ ]:


ax = sns.distplot(data_poisson,
                  kde=False,
                  color="skyblue",
                  hist_kws={"linewidth":15, "alpha":1})

ax.set(xlabel="Poisson Distribution", ylabel="Frequency")


# ### **Q. A warehouse typically receives 8 deliveries between 4 and 5 Friday.**
# 1. What is the probability that only 4 deliveries will arrive between 4 & 5pm on friday.
# 

# In[ ]:


poisson.pmf(4,8)


# 2. What is the probability of having less than 3 deliveries on friday

# In[ ]:


poisson.cdf(3,8)


# 3. What is the probability of having no delivery on friday between 4and 5 pm

# In[ ]:


poisson.pmf(0,8)


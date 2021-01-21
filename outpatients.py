
# coding: utf-8

# In[ ]:


from __future__ import division
import itertools
import numpy as np
import os.path
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


# In[2]:



def join_codes(row):
    return " ".join([str(v) for i, v in row.iteritems() if pd.notnull(v)])


# In[3]:


DATA_DIR = "C:\Users\enjie.zong\Desktop\CMS\SAMPLE1" #path to the file
EPSILON = 0.1


# In[4]:


# extract codes as bag of codes from input
opdf = pd.read_csv(
    os.path.join(DATA_DIR, "DE1_0_2008_to_2010_Outpatient_Claims_Sample_1.csv"),
    low_memory=False)
opdf.head()


# In[5]:


colnames = [colname for colname in opdf.columns if "_CD_" in colname]
bcdf = opdf.loc[:, colnames].apply(join_codes, axis=1)

# build a code-document matrix out of the codes
vec = CountVectorizer(min_df=1, binary=True)
X = vec.fit_transform(bcdf)


# In[8]:


sim = X.T * X


# In[ ]:


fout = open(os.path.join(DATA_DIR, "clusters.txt"), 'wb')
for row in range(0, X.shape[0]):
    codes = [code for code in X[row, :].nonzero()][1]
    dists = []
    for i, j in itertools.product(codes, codes):
        if i < j:
            sim_ij = sim.getrow(i).todense()[:, j][0]
            if sim_ij == 0:
                sim_ij = EPSILON
            dists.append(1 / (sim_ij ** 2)) 
    fout.write("%f\n" % (np.sqrt(sum(dists)) / len(dists)))
fout.close()


# In[ ]:


from livestats import livestats
import math
import os.path

EPSILON = 0.0001

def summary(stats):
    summary = {}
    summary["mean"] = stats.mean()
    summary["stdev"] = math.sqrt(stats.variance())
    q1, q2, q3 = [q[1] for q in stats.quantiles()]
    summary["q1"] = q1
    summary["q2"] = q2
    summary["q3"] = q3
    return summary    
    
def norm(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi)) *
        np.exp(-(x - mu)**2 / (2 * sigma**2)))

lns = 0
fin = open(os.path.join(DATA_DIR, "clusters.txt"), 'rb')
stats = livestats.LiveStats([0.25, 0.5, 0.75])
xs = []
for line in fin:
    line = line.strip()
    lns += 1
    x = EPSILON if line == "nan" else float(line)
    x = math.log(x, math.e)
    xs.append(x)
    # add a mirror image to make it approximately normal
#    xs.append(-x)
    stats.add(x)
#    stats.add(-x)
fin.close()

# plot data for visualization
mu = stats.mean()
sigma = math.sqrt(stats.variance())
count, bins, ignored = plt.hist(xs, bins=100, normed=True)
plt.plot(bins, [norm(x, mu, sigma) for x in bins], linewidth=2, color='r')
max_y = 0.5
#max_y = 10

# mean +/- (2*sigma or 3*sigma)
lb2 = mu - (2 * sigma)
ub2 = mu + (2 * sigma)
lb3 = mu - (3 * sigma)
ub3 = mu + (3 * sigma)
plt.plot([lb2, lb2], [0, max_y], 'r--')
plt.plot([ub2, ub2], [0, max_y], 'r--')
plt.plot([lb3, lb3], [0, max_y], 'r-')
plt.plot([ub3, ub3], [0, max_y], 'r-')

# median based (interquartile range based outlier measure)
q1, q2, q3 = [q[1] for q in stats.quantiles()]
iqr = q3 - q1
lb2 = q1 - (1.5 * iqr)
ub2 = q3 + (1.5 * iqr)
lb3 = q1 - (3.0 * iqr)
ub3 = q3 + (3.0 * iqr)
plt.plot([lb2, lb2], [0, max_y], 'g--')
plt.plot([ub2, ub2], [0, max_y], 'g--')
plt.plot([lb3, lb3], [0, max_y], 'g-')
plt.plot([ub3, ub3], [0, max_y], 'g-')

print summary(stats)

plt.show()


# In[ ]:


def compute_cutoff(level, cys):
    for i in range(len(cys), 0, -1):
        if cys[i-1] < level:
            return i
    return -1    
    
counts, bins, ignored = plt.hist(xs, bins=100)
cumsums = np.cumsum(counts)
plt.plot(bins[:-1], cumsums, color='red')

max_cy = len(xs)
strong_xcut = compute_cutoff(0.99 * max_cy, cumsums) / len(bins)
mild_xcut = compute_cutoff(0.95 * max_cy, cumsums) / len(bins)

print (strong_xcut, mild_xcut)

plt.plot([strong_xcut, strong_xcut], [0, max_cy], 'g-')
plt.plot([mild_xcut, mild_xcut], [0, max_cy], 'g--')

plt.show()


# In[ ]:


strong_outlier_cutoff = 0.9801980198019802
mild_outlier_cutoff = 0.36633663366336633

fin = open(os.path.join(DATA_DIR, "clusters.txt"), 'rb')
outliers = []
idx = 0
for line in fin:
    line = line.strip()
    x = EPSILON if line == "nan" else float(line)
    if x > mild_outlier_cutoff and x < 1.0:
       outliers.append((idx, x))
    idx += 1
fin.close()

# find corresponding claim ids and claims for verification
outliers_sorted = sorted(outliers, key=itemgetter(0), 
                         reverse=True)[0:10]
colnames = ["CLM_ID"]
colnames.extend([colname for colname in opdf.columns if "_CD_" in colname])
for idx, score in outliers_sorted:
    claim_id = opdf.ix[idx, colnames[0]]
    codes = opdf.ix[idx, colnames[1:]]
    names = ["_".join(x.split("_")[0:2]) for x in colnames[1:]]
    code_names = [":".join([x[0], x[1]]) for x in zip(names, codes.values) 
                                         if pd.notnull(x[1])]
    print("%s %6.4f %s" % (claim_id, score, str(code_names)))


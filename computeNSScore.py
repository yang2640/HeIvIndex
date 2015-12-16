import numpy as np

# load the query results
res = np.loadtxt('query.txt', dtype='float', usecols=(0,1,2,3))
nbr_images = len(res)

# compute score and return average
score = np.array([(res[i]//4)==(i//4) for i in xrange(nbr_images)]) * 1.0
average = np.sum(score) / (nbr_images)
print "average score:", average

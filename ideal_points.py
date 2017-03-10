import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib
from scipy.stats import norm
from joblib import Parallel, delayed
import plotly.plotly as py
import plotly.graph_objs as go
import datetime

icpsr_dict = {41: 'AL',81: 'AK',  61: 'AZ', 42: 'AR', 71: 'CA',62: 'CO',1: 'CT',11: 'DE', 43: 'FL', 44: 'GA', 82: 'HI',  
63: 'ID',21: 'IL', 22: 'IN', 31: 'IA',    32: 'KS',51: 'KY',45: 'LA', 2: 'ME',52: 'MD',3: 'MA',23: 'MI',33: 'MN',
46: 'MS',34: 'MO',64: 'MT',35: 'NE',65: 'NV',4: 'NH',12: 'NJ',66: 'NM',13: 'NY',47: 'NC',36: 'ND',24: 'OH',53: 'OK',
72: 'OR',14: 'PA',5: 'RI',48: 'SC',37: 'SD',54: 'TN',49: 'TX',67: 'UT',    6: 'VT', 40: 'VA', 73: 'WA', 56: 'WV', 
25: 'WI', 68: 'WY',55: 'DC'}

congress_num = 113
roll_call_link = 'ftp://voteview.com/dtaord/sen' + str(congress_num) + 'kh.ord'
roll_call_file = 'sen' + str(congress_num) + 'kh.ord'
data = urllib.urlretrieve(roll_call_link,roll_call_file) 

lines = open('sen' + str(congress_num) + 'kh.ord').readlines()
state_codes = []
names = []
votes = []
for line in lines[1:]:
    metadata = ""
    if len(line.split()[0]) == 8:
        snippet_string = line.split()[0] + '0' + line.split()[1] + line.split()[2]
    else:
        snippet_string = ''.join(line.split()[0:2])
    for idx in range(len(snippet_string)):
        if not snippet_string[idx].isdigit():
            break
        else:
            metadata = metadata + snippet_string[idx]
    name = ""
    switched = False
    started_chars = False
    started_name = False
    for idx in xrange(len(line)):
        if not line[idx].isdigit() and line[idx] != ' ' and not started_chars:
            started_chars = True
        elif line[idx].isdigit() and line[idx] != ' ' and started_chars and not switched:
            switched = True
        elif not line[idx].isdigit() and line[idx] != ' ' and started_chars and switched:
            name = name + line[idx]
            started_name = True
        elif line[idx].isdigit() and line[idx] != ' ' and started_name:
            break
    state_code = int(metadata[8:10])
    vote_snippet = line.split()[-1]
    char_location = -1
    for vote_idx in range(len(vote_snippet)):
        if not vote_snippet[vote_idx].isdigit():
            char_location = vote_idx
    vote = vote_snippet[(char_location+1):]
    state_codes.append(state_code)
    votes.append(vote)
    names.append(name)

d = {'state_code' : state_codes,
     'names' : names,
     'rollcall' : votes}
roll_df = pd.DataFrame(d)

roll_matrix = np.zeros((len(roll_df),len(roll_df.loc[0]['rollcall'])))
num_votes_in_row = np.zeros(len(roll_df))
for row_idx in range(len(roll_df)):
    num_votes = 0
    roll = roll_df.loc[row_idx]['rollcall']
    roll_array = np.zeros(len(roll))
    for roll_idx in range(len(roll)):
        if roll[roll_idx] in ['1','2','3']:
            roll_array[roll_idx] = 1
            num_votes = num_votes + 1
        elif roll[roll_idx] in ['4','5','6']:
            roll_array[roll_idx] = 0
            num_votes = num_votes + 1
        else:
            roll_array[roll_idx] = -1
    num_votes_in_row[row_idx] = num_votes
    roll_matrix[row_idx] = roll_array

roll_matrix = roll_matrix.astype(int)

roll_matrix = np.delete(roll_matrix, np.where(num_votes_in_row < 50)[0],0)
names = np.delete(np.array(names), np.where(num_votes_in_row < 50)[0])
state_codes = np.delete(np.array(state_codes), np.where(num_votes_in_row < 50)[0])

names = [name.title() for name in names]
ordered_states = np.array([icpsr_dict[state_code] for state_code in state_codes])
names = np.array([names[i] + ' (' + ordered_states[i] + ')' for i in range(len(ordered_states))])

y = roll_matrix
N = len(y)
J = len(y[0,])

mu_x = 0
mu_beta = np.zeros(2)
Sigma_x = 1
Sigma_beta = np.eye(2) * 25

Sigma_x_inv = 1/Sigma_x
Sigma_beta_inv = np.linalg.inv(Sigma_beta)

X = np.array([np.array([1.0,np.random.randn(1)]) for n in range(N)])
Beta = np.random.randn(J,2)
y_tilde = np.zeros((N,J))
posteriors = []

def E_step(i,J,X,Beta,y):
    row = np.zeros(J)
    for j in xrange(J):
        m_ij = np.dot(X[i,],Beta[j,])
        if y[i,j] == 1:
            row[j] = m_ij + norm.pdf(m_ij)/max(1e-16,norm.cdf(m_ij))
        elif y[i,j] == 0:
            row[j] = m_ij - norm.pdf(m_ij)/max(1e-16,1-norm.cdf(m_ij))
        else:
            row[j] = m_ij
    return row

start = datetime.datetime.now()
for it in range(500):
    first_term = np.sum(np.array([np.sum(np.array([np.dot(Beta[j,].T,X[i,]) ** 2 - 2*np.dot(Beta[j,],X[i,])*y_tilde[i,j] \
                    for j in range(J)])) for i in range(N)]))
    second_term = np.sum(np.array([X[i,1]**2*Sigma_x_inv-2*X[i,1]*mu_x*Sigma_x_inv for i in range(N)]))
    third_term = np.sum(np.array([np.dot(np.dot(Beta[j,],Sigma_beta_inv),Beta[j,])-\
                             2*np.dot(np.dot(Beta[j,],Sigma_beta_inv),mu_beta) for j in range(J)]))
    posterior = -1*first_term - second_term -third_term
    posteriors.append(posterior)
    print("Iteration: ", it)
    print("Posterior: ",posterior)
    y_tilde = np.array(Parallel(n_jobs=8)(delayed(E_step)(i,J,X,Beta,y) for i in range(N)))
    prev_X = np.copy(X[:,1])
    X[:,1] = np.array([(1/(Sigma_x_inv + np.sum(np.array([Beta[j,1]**2 for j in range(J)])))) *(
        Sigma_x_inv * mu_x + np.sum(np.array([Beta[j,1]*(y_tilde[i,j]-Beta[j,0]) for j in range(J)]))) for i in range(N)])
    change_X = np.linalg.norm(prev_X-X[:,1])
    print("Change in X", change_X)
    Beta = np.array([np.dot(np.linalg.inv(Sigma_beta_inv + np.sum(np.array([np.outer(X[i,],X[i,]) for i in range(N)]),axis=0)), \
        (np.dot(Sigma_beta_inv,mu_beta) + np.sum(np.array([X[i] * y_tilde[i,j] for i in range (N)]),axis=0))) for j in range(J)])
finish = datetime.datetime.now()
print(finish-start)

sorted_names = np.array([names[idx] for idx in np.argsort(X[:,1])])
sorted_X = np.sort(X[:,1])

sorted_names = np.array([names[idx] for idx in np.argsort(X[:,1])])
l = []
y = []
for i in range(int(N)):
    y.append((2000+i))
    trace0= go.Scatter(
        x= sorted_X[i],
        y= 0.0,
        mode= 'markers',
        marker= dict(size= 14,
                    line= dict(width=1),
                    color= 'hsl(0,500,500)',
                    opacity= 0.3
                   ),name= y[i],
        text= sorted_names[i]) 
    l.append(trace0);

layout= go.Layout(
    title= 'U.S. Senator Ideal Points, 2013-2015',
    hovermode= 'closest',
    autosize=False,
    width=1000,
    height=300,
    xaxis= dict(
        title= 'Ideal Point',
        range=[-2.8,4.3],
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig= go.Figure(data=l, layout=layout)
py.iplot(fig)

print sorted_names
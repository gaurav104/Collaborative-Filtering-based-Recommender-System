path = "ml-latest-small/"


# Importing libraries
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from tqdm import tqdm
import torch.nn.functional as F


# Reading the Data
ratings = pd.read_csv(path + 'ratings.csv')
movies = pd.read_csv(path + 'movies.csv')
min_rating,max_rating = ratings.rating.min(),ratings.rating.max()

u_uniq = ratings.userId.unique()
user2idx = {o:i for i,o in enumerate(u_uniq)}
ratings.userId = ratings.userId.apply(lambda x: user2idx[x])

m_uniq = ratings.movieId.unique()
movie2idx = {o:i for i,o in enumerate(m_uniq)}
ratings.movieId = ratings.movieId.apply(lambda x: movie2idx[x])

# Finding the number of users and movies
n_users = int(ratings.userId.nunique())
n_movies = int(ratings.movieId.nunique())

# Setting the embedding size
n_factors = 50


# Initializing the users and movies embedding matrix 
def get_emb(ni, nf):
	e = nn.Embedding(ni, nf)
	e.weight.data.uniform_(-0.01, 0.01)
	return e


class EmbeddingBias(nn.Module):
    def __init__(self, n_users, n_movies):
        super().__init__()

        (self.u, self.m, self.ub, self.mb) = [get_emb(*o) for o in [(n_users, n_factors), (n_movies, n_factors), (n_users,1), (n_movies, 1)]]
        '''
        self.u = nn.Embedding(n_users, n_factors)
        self.m = nn.Embedding(n_movies, n_factors)
        self.u.weight.data.uniform_(0,0.05)
        self.m.weight.data.uniform_(0,0.05)
        '''
    def forward(self, cats):
        users, movies = cats[:,0], cats[:,1]
        um = (self.u(users)*self.m(movies)).sum(1)
        res = um + self.ub(users).squeeze() + self.mb(movies).squeeze()
        res = F.sigmoid(res)*(max_rating-min_rating) + min_rating
        return res.view(-1,1)
    


#x = ratings.drop(['rating', 'timestamp'],axis=1)
#y = ratings['rating'].astype(np.float32)



# Creating the dataloader
class loader(Dataset):
	def __init__(self, path = "ml-latest-small/", transforms=None):
		self.path = path
		self.ratings = pd.read_csv(path + 'ratings.csv')
		self.ratings = shuffle(self.ratings)

		u_uniq = self.ratings.userId.unique()
		user2idx = {o:i for i,o in enumerate(u_uniq)}
		self.ratings.userId = self.ratings.userId.apply(lambda x: user2idx[x])

		m_uniq = self.ratings.movieId.unique()
		movie2idx = {o:i for i,o in enumerate(m_uniq)}
		self.ratings.movieId = self.ratings.movieId.apply(lambda x: movie2idx[x])

	def __getitem__(self, index):
		x = self.ratings.drop(['rating', 'timestamp'], axis=1).values
		y = self.ratings['rating'].values
		x, y  = torch.tensor(x), torch.tensor(y)
		x = x[index]
		y = y[index]

		return (x,y)

	def __len__(self):
		return (len(self.ratings))

# class for the purpose of loss calculation
class Average(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n

    #property
    def avg(self):
        return self.sum / self.count


# setting the hyper-parametes
batch_size = 64   
num_epochs = 2
lr = 0.01
wd=1e-5


train_set = loader()

# Building the train function
def train(learning_rate = 0.01, weight_decay = 1e-5):
	cuda = torch.cuda.is_available()
	net = EmbeddingBias(n_users, n_movies)
	if cuda:
		net = net.cuda()

	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate, weight_decay = weight_decay)
	
	print("Preparing training data...")
	train_loader = DataLoader(train_set, batch_size, shuffle=False)
	for epoch in tqdm(range(num_epochs)):
		train_loss = Average()
		net.train()
		for i, (x,y) in tqdm(enumerate(train_loader)):
			if cuda:
				x,y = x.cuda(), y.cuda()
			optimizer.zero_grad()
			outputs = net(x)

			loss = criterion(outputs.squeeze(), y.type(torch.float32))
			loss.backward()
			optimizer.step()
			train_loss.update(loss.item(), x.size(0))
			
		print("Epoch : {}/{}, Training Loss : {}".format(epoch+1, num_epochs, train_loss.avg()))
	return net


if __name__ == "__main__":
	train()





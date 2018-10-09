import numpy as np
from scipy.stats import logistic
from keras.datasets import mnist
import matplotlib.pyplot as plt
import time

def ReLU(x):
	a=np.zeros(x.shape) #initialise a of zeros
	ind=np.nonzero(x>0) #find index of nonzero x's
	a[ind]=x[ind] #set a to equal x for nonzero x's
	return a
def dReLU(x):
	a=np.zeros(x.shape) #initialise a with zeros
	ind=np.nonzero(x>0) #find index of nonzero x's
	a[ind]=1 #set a to equal 1 for nonzero x's
	return a
'''def lReLU(x): #leaky ReLU
	a=np.zeros(x.shape)
	ind=np.nonzero(x>0)
	a[ind]=x[ind]
	ind2=np.nonzero(x<=0)
	a[ind2]=0.01*x[ind2]
	return a
def dReLU(x): #leaky Relu
	a=np.zeros(x.shape)
	ind=np.nonzero(x>0)
	a[ind]=1
	ind2=np.nonzero(x<=0)
	a[ind2]=0.01
	return a
'''
def sigmoid(x):
#	a=1/(1+np.exp(-x))
	a=logistic.cdf(x) #sigmoid function
	return a
def dsigmoid(x):
	a=sigmoid(x)*(1-sigmoid(x)) #deriviative of sigmoid function
	return a	
'''
def cost(y,a2,nn): #def cost function
	W=0. #initialise W
	for layer in nn: #loop through layers
		W+=sum(sum(layer.w)) #set W to be sum of weights
	C=np.zeros((a2.shape))
	ind=np.nonzero(y==1)
	C[ind]=-np.log(a2[ind]+1e-9)
	ind2=np.nonzero(y==0)
	C[ind2]=-np.log(1-a2[ind2]+1e-9)
	C=np.sum(C)/a2.shape[0]
	print(C)
	return C
def dcost(y,a2):
	dC=np.zeros((a2.shape))
	ind=np.nonzero(y==1)
	dC[ind]=-1/(a2[ind]+1e-9)
	ind2=np.nonzero(y==0)
	dC[ind2]=1/(1-a2[ind2]+1e-9)
	return dC'''
def cost(y,a2,nn): #def cost function
	W=0. #initialise W
	for layer in nn: #loop through layers
		W+=sum(sum(layer.w)) #set W to be sum of weights
	C=np.zeros((a2.shape))
	for i,a in enumerate(y):
		C[i]=0.5*(a-a2[i])**2
	C=np.sum(C)/a2.shape[0]
	print(C)
	return C
def dcost(y,a2):
	dC=np.zeros((a2.shape))
	dC=a2-y
	return dC

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

class netlayer:
	#N size of previous layer
	#M size of currently layer
	def __init__(self,N,M):
		self.M=M
		self.N=N
		self.a1=np.zeros((M,1)) #initialise activation vector
		#self.w=np.zeros((M,self.N))
		self.b=0.01*np.ones((M,1)) #initialise bias vector
		self.w=np.random.randn(M,self.N)*np.sqrt(2/N) #initialise weights
#		self.w=np.ones((M,self.N))
	def fp(self,a0):
		self.a0=a0 #set self.a0 IS THIS NECESSARY?
		self.g=np.dot(self.w,a0) #calculate weights dot activations
		self.f=np.array([g + self.b for g in self.g[1]])#add in biases
		
		if self.layer==(Nlength-1): #if final layer apply sigmoid
			self.a1=sigmoid(self.f)
			return self.a1	
		else:
			self.a1=ReLU(self.f)#else apply ReLU

	def layer_location(self,l):
		self.layer=l
	def BP(self,a2):
			
		if self.layer==(Nlength-1):
			self.delta=(dcost(a2,self.a1)*dsigmoid(self.a1))
		
		else:	
			a2=np.transpose(a2,axes=[2,0,1])

			self.delta=a2*dReLU(self.a1) #calulate dc/da.dRelu/dz
		
				
			

		self.nbCt=np.array([delta @ a0.T for delta,a0 in zip(self.delta,self.a0)])			#calculate update to weights
		self.nbC=np.mean(self.nbCt,axis=0)
		self.dcda=np.array([np.dot(delta.T,self.w) for delta in self.delta]) #calculate dc/da for use in next layer
#		print('dcda')
#		print(self.dcda.shape)
		self.b+=learning_rate*np.mean(self.delta,axis=0)
#L2 reg		self.w-=learning_rate*(self.nbC+0.5*self.w) #update weights
		#update=-learning_rate*self.nbC
		#update_scale=np.linalg.norm(update.ravel())
		#param_scale=np.linalg.norm(self.w.ravel())
		#print('ratio: ',update_scale/param_scale)
		self.w=self.w-learning_rate*self.nbC

def forwardProp(neuralnet, inputdata):
	output=np.zeros((len(inputdata),neuralnet[len(neuralnet)-1].M,1))
#	for ind,data in enumerate(inputdata):
	neuralnet[0].fp(inputdata)
	for i in range(Nlength-2):
		neuralnet[i+1].fp(neuralnet[i].a1)
	output=neuralnet[Nlength-1].fp(neuralnet[Nlength-2].a1)
		
	return output

def backProp(neuralnet,datalabel):
#	for data in datalabel:
	neuralnet[Nlength-1].BP(datalabel)
	for i in (range(Nlength-2,-1,-1)):
		neuralnet[i].BP(neuralnet[i+1].dcda.T)
			

		
def learn(neuralnet, inputdata, datalabel,threshold,batch_size):	
	c0=1
	c1=threshold+2
	i=0
	num_Iter=1+len(inputdata)/batch_size
	costvec=[]
#	while abs(c0-c1)/c0>threshold:		
	for _ in range(1000):
#		t=time.time()
#		i+=1
		c0=c1
		xt,yt=next_batch(batch_size,inputdata,datalabel)
		output=forwardProp(neuralnet,xt)
		backProp(neuralnet, yt)
		output=forwardProp(neuralnet, inputdata)
		c1=cost(datalabel,output,neuralnet)
		costvec.append(c1)
#		print(time.time()-t)
#		die
#		print('iteration',i)
	return costvec

def classify(neuralnet,inputdata, datalabel):
	output=forwardProp(neuralnet, inputdata)
	correct=[]
	print(output.T)
	label=np.zeros((len(output),1))
	for i,out in enumerate(output):
		label[i]=np.argmax(out)
		#print('guess= ',label[i], ', value= ',datalabel[i],', correct? ',datalabel[i]==label[i])
		correct.append(datalabel[i]==label[i])
	classifcation_accuracy=sum(correct)/len(correct)*100.
	print(datalabel)
	print(label)
	print(classifcation_accuracy)
	

if __name__=='__main__':
#def run():
# Import data
	(a0,y_train),(X_test,y_test)=mnist.load_data()
#	ind=np.nonzero(y_train==1)
	a0=a0[:5000]

# Preprocess Data
	a0=a0-np.mean(a0) #zero centre training data
	a0/=np.std(a0) #normalise training data
	X_test=X_test-np.mean(X_test) #zero centre test data
	X_test=X_test/np.std(X_test)	#normalise test data
	y_train=y_train[:5000]
#	ind2=np.nonzero(y_test==1)
	test_data=X_test # limit number of text data
	test_data=np.reshape(test_data, (len(test_data),784,1)) #reshape rest
	test_labels=y_test # limit test labels
# format datalabels into binary array		
	a2=np.zeros((len(y_train),10,1)) #format training labels into binary vector
	for b,c in zip(a2,y_train): 
		b[c]=1
	a0=np.reshape(a0,(len(a0),784,1)) #reshape input data

	''' DEFINE PARAMETERS'''

	learning_rate=0.0	1 #define learning rate
	threshold=0.001	#define threshold
	batch_size=100 #define batch size for stochasitc gradient descent
	mshape=[16,16,10]
#	mshape=[300,10]	#define shape of NN
	Nlength=len(mshape)	#calculate number of layers
	nshape=[a0[0].size]	#initialize ashape
	nshape.extend(mshape[:len(mshape)-1])	#caluculate lengeth of weights
	''' INIT NN'''
	
	neuralnet=[netlayer(N=n,M=m) for n,m in zip(nshape,mshape)] #initialise neuralnetwork
	for i in range(Nlength):
		neuralnet[i].layer_location(i) #tell each layer where it is
	costvec=learn(neuralnet, a0, a2,threshold,batch_size) #learn
	plt.plot(costvec) #plot cost function over time
#	classify(neuralnet,a0,y_train)
	classify(neuralnet,a0[:10],y_train[:10])

	plt.show()


import theano
import theano.tensor as T
import numpy as np 
import matplotlib as plt 
def get_log_constant(log_unnorm_proba, x0):
	"""
	returns a compiled theano-function for log of normalization constant Z.
	According to MacKay:
	constant Z(x0) ~ P(x0)/sqrt(det 1/2pi Hess), where:
	x0 - peak of probablity
	Hess -Hessian of (  log_unnorm_proba).
	Therefore:
	log Z(x0) ~ log_unnorm_proba(x0)-0.5  ln(det 1/2pi Hess)

	:param: log_unnorm_proba: theano symbolic expression
	:param: x0: tensor or shared varaible for one sample in dataset
	:returns: theano symbolic expression for log Z
	"""
	t_det = T.nlinalg.Det() #theano constructor(!) of function for computation of determinant
	minus_Hess = -theano.gradient.hessian(log_unnorm_proba,wrt=x0)
	log_det_H = T.log(t_det(minus_Hess/2*np.pi))
	log_Z = log_unnorm_proba  - 0.5 * log_det_H
	return log_Z

if __name__=='__main__':
	"""
	example:
	let's make an autoencoder (5-2-5) for  gaussians 
	"""
	"""
	train_N = 1000
	#train_x in [-1;1] with high probability
	train_x = np.random.randn(train_N,5)
	
	X = T.matrix()
	encoder_params = theano.shared(np.random.randn(10))#10 params: 5x2 matrix
	decoder_params = theano.shared(np.random.randn(10))
	W_e = encoder_params.reshape((5,2))
	W_d = decoder_params.reshape((2,5))
	encode = T.tanh(T.dot(X, W_e))
	decode = T.tanh(T.dot( encode, W_d))
	error = T.mean(T.sqrt(T.sum((X - decode)**2, axis=1)))
	#training for 1000 iterations
	iter_num = 1000
	lr = 0.1
	grad = T.grad(error, [encoder_params, decoder_params])
	updates = ([(encoder_params,encoder_params - grad[0]*lr),(decoder_params,decoder_params - grad[1]*lr)]) 
	train = theano.function([X], error, updates= updates)
	monitor = theano.function([X], error)
	for i in xrange(0, iter_num):
		print 'params train', train(train_x)
	#now let's find x peak with fixed params
	x0 = theano.shared(np.random.randn(5))
	encode_per_vector =  T.tanh(T.dot(x0,W_e))
	decode_per_vector =  T.tanh(T.dot(encode_per_vector,W_d))
	unnormalized_log_proba = -T.sqrt(T.sum(x0 - decode_per_vector)**2)
	grad2 = T.grad(unnormalized_log_proba, x0)
	lr2 = 0.0001
	updates2 = ([(x0, x0+lr2*grad2)]) 

	train2 = theano.function([], unnormalized_log_proba, updates= updates2)
	
	
	logZ = get_log_constant(unnormalized_log_proba, x0)
	logZ_function = theano.function( [],logZ)
	#note, we can skip the minimum due to optimization problems or non-convex function
	found_const = False
	found_const_iter = 0
	while not found_const:
		best_peak = None
		best_proba = -9999
		found_const_iter+=1
		for i in xrange(0, iter_num):
			current_proba = train2()
			if current_proba>best_proba:
				best_peak = x0.eval()
				best_proba = current_proba
			
			print 'peak find', best_proba, current_proba, found_const_iter
			x0.set_value(best_peak)
			#constant calculation
			
		
		result = logZ_function()
		if np.isnan(result):
			print 'NOT FOUND'
			x0.set_value(np.random.randn(5))
		else:
			print result
			found_const = True	
	"""
	"""
	Another example: 
	1.let's consider a simple array of 1d gaussians (virtually, we don't need to generate this during the example) with sigma_sqr = 0.5
	It's probability is: 1/sqrt(2pi) exp(-x**2)
	2.Let's unnormalizaed  log proba be: log p(x) = -x**2
	3.expected const is  (1/sqrt(2*sigma_sqr*pi)) = sqrt (1/(pi))
	"""
	
	x0 = theano.shared(np.zeros(1))#peak
	unnorm_log_proba = -x0**2#note: it's a vector, we need scalar. So, use unnorm_log_proba[0] below
	log_Z = get_log_constant(unnorm_log_proba[0], x0)
	logZ_function = theano.function( [],log_Z)
	print np.exp(logZ_function())
	print 'expected',  np.sqrt(1/np.pi)

	
		


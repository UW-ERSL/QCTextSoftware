"""
Created on Tue Jul 16 05:23:29 2024

@author: Krishnan Suresh; ksuresh@wisc.edu
"""
import numpy as np
import scipy
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit import QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import  HamiltonianGate
from qiskit.circuit.library import QFT
from qiskit.circuit.library.standard_gates.u import UGate
import matplotlib.pyplot as plt

class HHLQiskit:
	def __init__(self, A, b, f = 0.5, lambdaHat = 1,
			  m = 3,nQPEShots = 50,nHHLShots = 1000,debug = False):	
		self.A = A
		self.bNorm = scipy.linalg.norm(b) # store for later
		self.b = b/self.bNorm# normalize b, so it is ready for loading.
		self.f = f # Used in Hamiltonian evolution. 
		self.lambdaHat = lambdaHat # Used in Hamiltonian evolution. 
		self.m = m # Number of bits to estimate the eigenphasess
		self.nQPEShots = nQPEShots
		self.nHHLShots = nHHLShots
		self.N = self.A.shape[0]
		self.n = int(np.log2(self.N))
		self.dataOK = True
		if np.abs(2**self.n - self.A.shape[0]) > 1e-10: 
			print("Invalid size of matrix; must be power of 2") 
			self.dataOK = False
		if (self.n > 4):
			print('Matrix size is too large')
			self.dataOK = False
		symErr = np.max(self.A - np.transpose(self.A))
		if (symErr > 0):
			print('A does not appear to be symmetric')
			self.dataOK = False
		if (m < 1):
			print('m has to be at least 1')
			self.dataOK = False
		if not (A.shape[0] == b.shape[0]):
			print('A and b sizes are not compatible')
			self.dataOK = False
			
		self.debug = debug
	
	def solveExact(self):
		self.x_exact = scipy.linalg.solve(self.A, self.bNorm*self.b)

	def computeEigen(self):
		# used for verification and not part of HHL
		self.eig_val, self.eig_vec = scipy.linalg.eig(self.A)
		# Since A is assumed to be symmetric the eigen values are real
		# So get rid of the imaginary component
		self.eig_val = np.abs(self.eig_val)
		
	def simulateCircuit(self,circuit,nShots=1000):
		backend = Aer.get_backend('qasm_simulator')
		new_circuit = transpile(circuit, backend)
		job = backend.run(new_circuit,shots = nShots)
		counts = job.result().get_counts(circuit)
		return counts

	def processQPECounts(self,counts):
		# Utility function for QPE
		# First sort descending using 2nd item in dictionary
		self.QPECountsSorted = sorted(counts.items(),
			key=lambda item: item[1],reverse=True)
		m = len(self.QPECountsSorted[0][0]) # length of bit string
		values = []
		for i in range(len(self.QPECountsSorted)):
			string = self.QPECountsSorted[i][0]
			values.append(int(string, 2)/(2**m))
	
		self.thetaTilde = np.array(values)
		self.lambdaTilde = self.thetaTilde*self.lambdaHat/self.f
		return 
	
	def constructQPECircuit(self,mode):
		phase_qubits = QuantumRegister(self.m, '\u03B8')
		b_qubits = QuantumRegister(self.n, 'b')
		if (mode == 'HHL') or (mode == 'IQPE'):
			ancilla_qubit = QuantumRegister(1,'a')
			offset = 1
			cl = ClassicalRegister(1+self.n,'cl')
			circuit = QuantumCircuit(ancilla_qubit, phase_qubits, b_qubits,cl)	
		elif (mode == 'QPE'):
			offset = 0
			cl = ClassicalRegister(self.m,'cl')
			circuit = QuantumCircuit( phase_qubits, b_qubits,cl)	
		else:
			print('Incorrect mode in constructQPECircuit')
			return
			
		for i in range(self.m):
			circuit.h(offset+i)
		if (mode == 'QPE') or (mode == 'HHL'):
			circuit.prepare_state(Statevector(self.b),[*range(self.m+offset, self.n+self.m+offset)],'b')
		t = -2*np.pi*self.f/self.lambdaHat #Note negative
		U_A = HamiltonianGate(self.A, time=t,label = 'UA')
		U_A._name = 'UA'
		for i in range(self.m):
			U_A_pow = U_A.power(2**i) 
			UControl = U_A_pow.control(1) # only 1 control qubit
			circuit.append(UControl,[i+offset,*range(self.m+offset, self.n+self.m+offset)])
		iqft = QFT(num_qubits=self.m,inverse=True).to_gate()
		iqft._name = 'IQFT'
		circuit.append(iqft, [*range(offset,self.m+offset)])
		return circuit
	
	def constructHHLCircuit(self):
		self.HHLCircuit = self.constructQPECircuit('HHL')
		self.HHLCircuit.barrier()
		nControlledRotations = min(len(self.QPECountsSorted),self.N)
		for i in range(nControlledRotations):
			phaseString = self.QPECountsSorted[i][0] 
			alpha = 2*np.arcsin(self.C/self.lambdaTilde[i])		
			cu_gate = UGate(alpha, 0, 0).control(m, ctrl_state = phaseString) 
			self.HHLCircuit.append(cu_gate,[*range(1,1+m),0])
		qpeInverse = self.constructQPECircuit('IQPE').inverse()
		self.HHLCircuit.barrier()
		self.HHLCircuit.compose(qpeInverse,[*range(0,1+self.m+self.n)], [*range(0,1+self.n)],inplace = True)
		self.HHLCircuit.barrier()
		self.HHLCircuit.measure([0,*range(1+self.m,1+self.m+self.n)], [0,*range(0,self.n)]) 
		return 
		
	def computeCompliance(self):
		return np.dot(self.b,self.x_hhl)

	def computeExactCompliance(self):
		return np.dot(self.b,self.x_exact)
	
	def scale_x_hhl(self):
		bComputed = np.matmul(self.A,self.x_hhl)
		scale =  self.bNorm/np.linalg.norm(bComputed)
		for i in range(self.x_hhl.shape[0]):
			self.x_hhl[i] = self.x_hhl[i]*scale
			
	def execute(self):
		if not self.dataOK:
			print('Check input data')
			return False

		## Step 1: Construct  QPE circuit and estimate theta
		self.QPECircuit = self.constructQPECircuit('QPE')
		self.QPECircuit.measure([*range(0,self.m)], [*range(0,self.m)]) 
		counts = self.simulateCircuit(self.QPECircuit,self.nQPEShots)
		self.processQPECounts(counts)
		print('\u03B8 estimate: ', self.thetaTilde)
		print('\u03BB estimate: ', self.lambdaTilde)
		
		# compute C for conditional rotation
		self.C = 0.99*min(self.lambdaTilde)
		
		self.constructHHLCircuit()
		
		HHLCounts = self.simulateCircuit(self.HHLCircuit,self.nHHLShots)
		print(HHLCounts)
		
		return True
#################################
if __name__ == '__main__':
	# For all examples, note that
	# 1. f should be < 1.
	# 2. Estimate of lambdaTilde is needed for Hamiltonian evolution. 
	# 2. m controls the accuracy of representing eigen phase
	# 3. nQPEShots is the number of QPE measurements
	# 4. nHHLShots is the number of HHL measurements

	
	example = 1
	debug = False
	nQPEShots = 50
	nHHLShots = 1000
	if (example == 1):
		A = np.array([[1,0],[0,0.75]])
		b = np.array([np.sqrt(3)/2,0.5])
		f = 0.5
		lambdaHat = 1
		m = 2
	elif (example == 2):
		A = np.array([[2,-1],[-1,2]])
		b = np.array([np.sqrt(3)/2,0.5])
		f = 0.5
		lambdaHat = 1
		m = 2
	elif (example == 3):
		A = np.array([[1,0,0,-0.5],[0,1,0,0],[0,0,1,0],[-0.5,0,0,1]])
		b = np.array([1,1,1,1])/np.sqrt(4)
		f = 0.5
		lambdaHat = 1
		m = 2
	elif (example == 4):
		A = np.array([[2,-1,0,0],[-1,2,-1,0],[0,-1,2,-1],[0,0,-1,2]])
		b = np.array([0,1])
		f = 0.5
		lambdaHat = 1
		m = 3

	plt.close('all')
	# Create HHL object	
	HHL = HHLQiskit(A,b,f=f,lambdaHat = lambdaHat,
				 m = m, nQPEShots = nQPEShots,
				 nHHLShots = nHHLShots, debug = debug)
	
	# Execute main code
	HHL.execute()
	
	HHL.QPECircuit.draw('mpl')
	HHL.HHLCircuit.draw('mpl')
	
	if (debug):# if debug
		HHL.solveExact();
		print("Exact sol A:\n", HHL.x_exact)
		HHL.computeEigen()
		print("Exact eigen values of A:\n", HHL.eig_val)
		print("Exact eigen vectors of A:\n", HHL.eig_vec)
		eigenPhase = HHL.f*HHL.eig_val/HHL.lambdaHat
		tMax = 2*np.pi/max(HHL.eig_val)
		print("Exact eigenphases of A:\n", eigenPhase)
		if (max(eigenPhase) >= 1):
			print('Invalid value for lambdaHat and/or fs', HHL.lambdaHat,HHL.f)
			print('Here choose t to be <', abs(tMax))
			
		print('x_exact:', HHL.x_exact)
		print('x_HHL:', HHL.x_hhl)
		fidelity = np.dot(HHL.x_hhl, HHL.x_exact)/(np.linalg.norm(HHL.x_hhl)*np.linalg.norm(HHL.x_exact))
		print('fidelity:', fidelity)
		
		print('Est Compliance:', HHL.computeCompliance())
		print('Exact Compliance:', HHL.computeExactCompliance())
	
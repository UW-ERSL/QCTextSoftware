# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 12:16:44 2024

@author: Krishnan Suresh; ksuresh@wisc.edu
"""


#%% Modules needed for all examples below
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from IPython.display import display
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import HamiltonianGate
from qiskit.circuit.library import QFT, PhaseEstimation
from qiskit_aer import Aer
import matplotlib.pyplot as plt
import numpy as np


#%%  Sampling a circuit on a simulator
def simulateCircuit(circuit,nShots=1000):
	backend = Aer.get_backend('qasm_simulator')
	new_circuit = transpile(circuit, backend)
	job = backend.run(new_circuit,shots = nShots)
	counts = job.result().get_counts(circuit)
	return counts


#%% Hamiltonian
A = np.array([[2,-1],[-1,2]])
f = 0.5
lambdaHat = 3
t = -2*np.pi*f/lambdaHat #Note negative
U_A = HamiltonianGate(A, time=t,label = 'UA')
print(np.array(U_A.to_matrix()))
v = np.array([1/np.sqrt(2),-1/np.sqrt(2)])
circuit = QuantumCircuit(1)
circuit.prepare_state(Statevector(v) ,0,'Prepare v')
circuit.append(U_A, qargs=[0])
circuit.draw('mpl') 
psi = Statevector(circuit)
display(psi.draw('latex'))


#%% Single digit QPE with single qubit v
def myQPE1(A,v,f,lambdaHat):
	circuit = QuantumCircuit(2,1)
	circuit.h(0)
	circuit.prepare_state(Statevector(v),[1],' v')
	t = -2*np.pi*f/lambdaHat #Note negative
	U_A = HamiltonianGate(A, time=t,label = 'UA')
	UControl = U_A.control(1)
	circuit.append(UControl,[0,1])
	iqft = QFT(num_qubits=1,inverse=True).to_gate()
	iqft._name = 'IQFT'
	circuit.append(iqft, [0])
	circuit.measure([0], [0]) 
	circuit.draw('mpl') 
	counts = simulateCircuit(circuit,nShots=1)
	return counts

#%% Single digit QPE with multiple qubits v
def myQPE2(A,v,f,lambdaHat):
	n = int(np.log2(v.shape[0]))
	circuit = QuantumCircuit(n+1,1)
	circuit.h(0)
	circuit.prepare_state(Statevector(v),[*range(1, n+1)],'v')
	t = -2*np.pi*f/lambdaHat #Note negative
	U_A = HamiltonianGate(A, time=t,label = 'UA')
	UControl = U_A.control(1) # only 1 control qubit
	circuit.append(UControl,[*range(0, n+1)])
	iqft = QFT(num_qubits=1,inverse=True).to_gate()
	iqft._name = 'IQFT'
	circuit.append(iqft, [0])
	circuit.measure([0], [0]) 
	return simulateCircuit(circuit,nShots=1)


#%% Multiple digit QPE with multiple qubits v
def myQPE3(A,v,f,lambdaHat,m=1):
	N = v.shape[0]
	n = int(np.log2(N))
	phase_qubits = QuantumRegister(m, '\u03B8')
	input_qubits = QuantumRegister(n, 'b')
	phase_measurements = ClassicalRegister(m, '\u0398')
	circuit = QuantumCircuit(phase_qubits,input_qubits,phase_measurements)
	for i in range(m):
		circuit.h(i)
	circuit.prepare_state(Statevector(v),[*range(m, n+m)],'b')
	t = -2*np.pi*f/lambdaHat #Note negative
	U_A = HamiltonianGate(A, time=t,label = 'UA')
	U_A._name = 'UA'
	for i in range(m):
		U_A_pow = U_A.power(2**i) 
		UControl = U_A_pow.control(1) # only 1 control qubit
		circuit.append(UControl,[i,*range(m, n+m)])
	iqft = QFT(num_qubits=m,inverse=True).to_gate()
	iqft._name = 'IQFT'
	circuit.append(iqft, [*range(0,m)])
	circuit.measure([*range(0,m)], [*range(0,m)]) 
	circuit.draw('mpl') 
	counts = simulateCircuit(circuit,nShots=50)
	return counts

#%% Utility function processCounts for QPE
def processCounts(counts):
	# Input:  counts from circuit simulation 
	# Return: decimal values sorted by descending probability
	# First sort descending using 2nd item in dictionary

	countsSorted = sorted(counts.items(),
		key=lambda item: item[1],reverse=True)
	m = len(countsSorted[0][0]) # length of bit string
	values = []
	for i in range(len(countsSorted)):
		string = countsSorted[i][0]
		values.append(int(string, 2)/(2**m))
	return np.array(values)


#%% Test cases for QPE
plt.close('all')
example = 2
if (example == 1):
	A = np.array([[1,0],[0,0.75]])
	v0 = np.array([1,0])
	v1 = np.array([0,1])
	a0 = 1/2
	a1 = np.sqrt(3)/2
	v = a0*v0 +  a1*v1
	f = 0.5
	lambdaHat = 1
	m = 2
elif (example == 2):
	A = np.array([[2,-1],[-1,2]])
	v0 = np.array([1/np.sqrt(2),1/np.sqrt(2)])
	v1 = np.array([1/np.sqrt(2),-1/np.sqrt(2)])
	a0 = 1
	a1 = 0
	v = a0*v0 +  a1*v1
	f = 0.5
	lambdaHat = 3
	m = 2
elif (example == 3):
	A = np.array([[1,0,0,-0.5],[0,1,0,0],[0,0,1,0],[-0.5,0,0,1]])
	v0 = np.array([1/np.sqrt(2),0,0,-1/np.sqrt(2)])
	v1 = np.array([1/np.sqrt(2),0,0,1/np.sqrt(2)])
	v2 = np.array([0,1,0,0])
	v3 = np.array([0,0,1,0])
	a = [1/np.sqrt(4),1/np.sqrt(4),1/np.sqrt(4),1/np.sqrt(4)]
	#a = [1,0,0,0]
	v = a[0]*v0 + a[1]*v1 + a[2]*v2 + a[3]*v3
	f = 0.5
	lambdaHat = 1.5
	m = 2
elif (example == 4):
	A = np.array([[2,-1,0,0],[-1,2,-1,0],[0,-1,2,-1],[0,0,-1,2]])
	v = np.random.rand(4)
	v = v/np.linalg.norm(v)
	f = 0.5
	lambdaHat = 4
	m = 2

counts = myQPE3(A,v,f,lambdaHat,m=m)	
print("counts:", counts)
thetaTilde = processCounts(counts)
print("thetaTilde:", thetaTilde)
print("EigenvalueTilde:",thetaTilde*lambdaHat/f)

#%% Using the Built in QPE
m = 3
A = np.array([[2,-1,0,0],[-1,2,-1,0],[0,-1,2,-1],[0,0,-1,2]])
v = np.random.rand(4)
v = v/np.linalg.norm(v)
f = 0.5
lambdaHat = 4

t = -2*np.pi*f/lambdaHat #Note negative
U_A = HamiltonianGate(A, time=t,label = 'UA')
iqft = QFT(num_qubits=m,inverse=True).to_gate()
iqft._name = 'IQFT'
qpe = PhaseEstimation(m,U_A,iqft)

#%% Pauli Expansion 2x 2
def PauliExpansion2x2(A):
	I = np.array([[1,0],[0,1]])
	X = np.array([[0,1],[1,0]])
	Y = np.array([[0,-1j],[1j,0]])
	Z = np.array([[1,0],[0,-1]])
	a = 4*[0]
	a[0] = np.trace(np.matmul(A,I))/2
	a[1] = np.trace(np.matmul(A,X))/2
	a[2] = np.trace(np.matmul(A,Y))/2
	a[3] = np.trace(np.matmul(A,Z))/2
	return a

A = np.array([[2,-1],[-1,2]])
a= PauliExpansion2x2(A)
print(a)

#%% Pauli Expansion 4 x 4
def PauliExpansion4x4(A):
	I = np.array([[1,0],[0,1]])
	X = np.array([[0,1],[1,0]])
	Y = np.array([[0,-1j],[1j,0]])
	Z = np.array([[1,0],[0,-1]])
	basis2x2 = [I,X,Y,Z]
	basis4x4 = 16*[None]
	count = 0
	for i in range(4):
		for j in range(4):
			basis4x4[count] = np.kron(basis2x2[i],basis2x2[j])
			count = count+1
	a = 16*[0]
	for i in range(16):
		a[i] = np.trace(np.matmul(A,basis4x4[i]))/4

	return a

A = np.array([[1,0,0,-0.5],[0,1,0,0],[0,0,1,0],[-0.5,0,0,1]])
a= PauliExpansion4x4(A)
print(a)

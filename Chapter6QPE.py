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
from qiskit.circuit.library import QFT, PhaseEstimation, UnitaryGate
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


#%% Schematic
zeroQubit = QuantumRegister(1, '0')
vQubit = QuantumRegister(1, 'v')
circuit = QuantumCircuit(zeroQubit,vQubit) 
circuit.h(0)
UMatrix = 1/np.sqrt(2)*np.array([[1,1],[1j,-1j]]) 
U = UnitaryGate(UMatrix,'U_A')
UControl = U.control(1)
circuit.append(UControl,[0,1])
circuit.draw('mpl') 


#%% Single digit QPE with multiple qubits v
def myQPESingleBit(A,v,lambdaHat,nShots=1000):
	n = int(np.log2(v.shape[0]))
	circuit = QuantumCircuit(n+1,1)
	circuit.h(0)
	circuit.prepare_state(Statevector(v),[*range(1, n+1)],'v')
	t = -2*np.pi/lambdaHat #Note negative
	U_A = HamiltonianGate(A, time=t,label = 'UA')
	UControl = U_A.control(1) # only 1 control qubit
	circuit.append(UControl,[*range(0, n+1)])
	iqft = QFT(num_qubits=1,inverse=True).to_gate()
	iqft._name = 'IQFT'
	circuit.append(iqft, [0])
	circuit.measure([0], [0]) 
	counts = simulateCircuit(circuit,nShots)
	probabilities = np.array([])
	thetaEstimates = np.array([])
	countsSorted = {k: v for k, v in sorted(counts.items(), 
										 key=lambda item: item[1],
										 reverse=True)}
	for key in countsSorted:
		probabilities = np.append(probabilities,countsSorted[key]/nShots)
		thetaEstimates = np.append(thetaEstimates,int(key, 2)/(2))
	return [thetaEstimates,probabilities]

#%% Multiple digit QPE with multiple qubits v
def myQPEMultiBit(A,v,lambdaHat,m,nShots=1000):
	N = v.shape[0]
	n = int(np.log2(N))
	phase_qubits = QuantumRegister(m, '\u03B8')
	input_qubits = QuantumRegister(n, 'b')
	phase_measurements = ClassicalRegister(m, '\u0398')
	circuit = QuantumCircuit(phase_qubits,input_qubits,phase_measurements)
	for i in range(m):
		circuit.h(i)
	circuit.prepare_state(Statevector(v),[*range(m, n+m)],'b')
	t = -2*np.pi/lambdaHat #Note negative
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
	#circuit.draw('mpl') 
	counts = simulateCircuit(circuit,nShots)
	countsSorted = {k: v for k, v in sorted(counts.items(), 
										 key=lambda item: item[1],
										 reverse=True)}
	probabilities = np.array([])
	thetaEstimates = np.array([])
	for key in countsSorted:
		probabilities = np.append(probabilities,countsSorted[key]/nShots)
		thetaEstimates = np.append(thetaEstimates,int(key, 2)/(2**m))
	return [thetaEstimates,probabilities]



#%% Test cases for QPE
plt.close('all')
example = 1
if (example == 1):
	A = np.array([[1,0],[0,0.75]])
	v0 = np.array([0,1])
	v1 = np.array([1,0])
	Lambda = [0.75,1]
	a = [1/np.sqrt(2),1/np.sqrt(2)]
	a= [1,0]
	v = a[0]*v0 + a[1]*v1
	lambdaHat = 2
	m = 3
elif (example == 2):
	A = np.array([[2,-1],[-1,2]])
	v0 = np.array([1/np.sqrt(2),1/np.sqrt(2)])
	v1 = np.array([1/np.sqrt(2),-1/np.sqrt(2)])
	Lambda = [1,3]
	a = [1,0]
	v = a[0]*v0 + a[1]*v1
	lambdaHat = 6
	m = 10
elif (example == 3):
	A = np.array([[1,0,0,-0.5],[0,1,0,0],[0,0,1,0],[-0.5,0,0,1]])
	v0 = np.array([1/np.sqrt(2),0,0,1/np.sqrt(2)])
	v1 = np.array([0,1,0,0])
	v2 = np.array([0,0,1,0])
	v3 = np.array([1/np.sqrt(2),0,0,-1/np.sqrt(2)])
	Lambda = [0.5,1,1,1.5]
	#a = [1/np.sqrt(4),1/np.sqrt(4),1/np.sqrt(4),1/np.sqrt(4)]
	a = [0,0,1,0]
	v = a[0]*v0 + a[1]*v1 + a[2]*v2 + a[3]*v3
	lambdaHat = 3
	m = 10

[thetaEstimates,P] = myQPEMultiBit(A,v,lambdaHat,m)	
print("thetaEstimates:",thetaEstimates)
print("probabilities:", P)
thetaTilde = np.sum(thetaEstimates*P)
print("thetaTilde:", thetaTilde)
print("EigenvalueTilde:",thetaTilde*lambdaHat)

#%% Using the Built in QFT and PhaseEstimation
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

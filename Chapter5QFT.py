# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 05:23:29 2024

@author: Krishnan Suresh; ksuresh@wisc.edu
"""
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from IPython.display import display
from qiskit.quantum_info import Statevector, Operator
from qiskit.circuit.library import QFT
from qiskit.visualization import plot_histogram
from qiskit_aer import Aer
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision =3,suppress=True)

plt.close('all')

def simulateCircuit(circuit,nShots=1000):
	backend = Aer.get_backend('aer_simulator')
	new_circuit = transpile(circuit, backend)
	job = backend.run(new_circuit,shots = nShots)
	counts = job.result().get_counts(circuit)
	return counts

#%% genertate signal
def continuousSignal(t):
	example = 3
	if (example == 1):
		s = 0.5*np.sin(2*2*np.pi*t) 
	elif (example == 2):
		s = 0.5*np.sin(2*2*np.pi*t) - 0.3*np.cos(5*2*np.pi*t)
	else:
		s = 0.25 + 0.5*np.sin(2*2*np.pi*t) - 0.3*np.cos(5*2*np.pi*t)
	return s

nContinuousSamples = 1000 # for plotting
tPlot = np.linspace(0,1,nContinuousSamples,endpoint = False)
yPlot = continuousSignal(tPlot)

M = 32 # We set the number of discrete samples here
t = np.linspace(0,1,M,endpoint = False) # need to eliminate the last point
y = continuousSignal(t)

#%% Plotting of signals
plt.plot(tPlot,yPlot)
plt.axhline(0, color='black')
plt.xlabel('Time', fontsize=14)
plt.ylabel('Signal', fontsize=14)
plt.grid(visible=True)


plt.figure()
plt.axhline(0, color='black')
plt.xlabel('Time', fontsize=14)
plt.ylabel('Signal', fontsize=14)
plt.grid(visible=True)
plt.plot(t,y,'*')


#%% DFT operation and processing
def createDFTMatrix(M):
    DFTMatrix = np.zeros((M,M), dtype=complex)
    omega = np.exp(1j*(2*np.pi/M))
    for i in range(M):
        for j in range(M):
            DFTMatrix[i][j] = omega**(-i*j)
    return DFTMatrix 

M = len(y)
DFTMatrix = createDFTMatrix(M)
phi = np.matmul(DFTMatrix,y)
# Process DFT Result
def processDFTResult(phi):
	M = len(phi)
	phi = phi/M
	c0 =  phi[0].real # constant
	a = (phi[1:int(M/2)]).real+(phi[M-1:int(M/2):-1]).real; # cosine terms
	b =  (phi[M-1:int(M/2):-1] - phi[1:int(M/2)]).imag; # sine terms
	b = np.insert(b, 0,0)
	return [c0,a,b]

[c0,a,b] = processDFTResult(phi)

plt.figure()
plt.bar(0,c0, label =r"$c_0$")
plt.bar(list(range(1,int(M/2))),a, label =r"$a_i$")
plt.bar(list(range(0,int(M/2))),b, label =r"$b_i$")
plt.legend( fontsize=14)
plt.axhline(0, color='black')
plt.xlabel('Frequency', fontsize=14)
plt.ylabel('Amplitude', fontsize=14)


#%% QFT-2 
circuit = QuantumCircuit(2, 2)  
circuit.h(1)
circuit.cp(np.pi/2,0,1) 
circuit.h(0) 
circuit.swap(0,1)
print(np.array(Operator(circuit).data))
circuit.measure([0,1], [0,1]) 
circuit.draw('mpl') 
counts = simulateCircuit(circuit,1000)
print('Counts:',counts)

#%% QFT-3
m = 3 # number of qubits
circuit = QuantumCircuit(m, m) 
qft = QFT(num_qubits=m).to_gate()
circuit.append(qft, qargs=list(range(m)))
circuit.measure(list(range(m)),list(range(m))) 
circuit.decompose(reps=2).draw('mpl') 

#%% QFT circuit from scratch
def myQFT(m): # m is the # of qubits
    q = QuantumRegister(m, 'q')
    c = ClassicalRegister(m,'c')
    circuit = QuantumCircuit(q,c)
    for k in range(m):
        j = m - k
        circuit.h(q[j-1])
        circuit.barrier()
        for i in reversed(range(j-1)):
            circuit.cp(2*np.pi/2**(j-i),q[i], q[j-1])
      
    circuit.barrier()  
    for i in range(m//2):
        circuit.swap(q[i], q[m-i-1])
    return circuit

m = 3
circuit = myQFT(m)
circuit.measure(list(range(m)),list(range(m))) 
circuit.draw('mpl') 

#%% Verify QFT Matrix
def createQFTMatrix(M):
    QFTMatrix = np.zeros((M,M), dtype=complex)
    omega = np.exp(2*np.pi*1j/M)
    for i in range(M):
        for j in range(M):
            QFTMatrix[i][j] = omega**(i*j)
    return QFTMatrix/np.sqrt(M)

m = 3
circuit = myQFT(m)
UFromCircuit = Operator(circuit)
UExact = createQFTMatrix(2**m)
print("Error: ", round(np.linalg.norm(UFromCircuit-UExact),10))

#%% QFT signal processing
def QFTSignalProcessing(y,nShots):
	M = len(y) # length of signal
	m = int(np.log2(M)) # number of qubits
	circuit = QuantumCircuit(m, m)  
	q = Statevector(y/np.linalg.norm(y)) 
	circuit.prepare_state(q,list(range(m)),'Prepare q')
	qftCircuit = myQFT(m)
	circuit.append(qftCircuit, qargs=list(range(m)),cargs=list(range(m)))
	circuit.measure(list(range(m)),list(range(m))) 
	counts = simulateCircuit(circuit,nShots=nShots)
	return counts

nShots = 1000
counts = QFTSignalProcessing(y,nShots)
plot_histogram(counts)

#%% Process QFT resuts
def processQFTResult(counts,nShots):
	phi = np.zeros(M)
	for i in counts:
		freq = int(i, 2)
		phi[freq] = np.sqrt(counts[i]/nShots)
	ampl = (phi[1:int(M/2)])+(phi[M-1:int(M/2):-1]);
	ampl = np.insert(ampl, 0,phi[0].real)
	return ampl
	
ampl = processQFTResult(counts,nShots)

plt.figure()
plt.bar(list(range(0,int(M/2))),ampl)
plt.axhline(0, color='black')
plt.xlabel('Frequency', fontsize=14)
plt.ylabel('Amplitude', fontsize=14)

#%% QFT signal processing
M = len(y) # length of signal
m = int(np.log2(M)) # number of qubits
circuit = QuantumCircuit(m, m)  
yNormalized = y/np.linalg.norm(y)
q = Statevector(yNormalized) 
circuit.prepare_state(q,list(range(m)),'Prepare q')
qft = QFT(num_qubits=m).to_gate()
circuit.append(qft, qargs=list(range(m)))
circuit.measure(list(range(m)),list(range(m))) 
nShots = 1000
counts = simulateCircuit(circuit,nShots=nShots)
print('Counts:',counts)
plot_histogram(counts)
qftResult = np.zeros(2**(m-1))
for i in counts:
	freq = int(i, 2)
	if (freq > (2**m)/2):
		freq = (2**m)-freq
	ampl = counts[i]/nShots
	qftResult[freq] = qftResult[freq] + ampl

print(qftResult)
#%% QFT-3
m = 3 # number of qubits
circuit = QuantumCircuit(m, m) 
circuit.x(0)
circuit.x(1)
circuit.x(2)
qft = QFT(num_qubits=m).to_gate()
circuit.append(qft, qargs=list(range(m)))
psi = Statevector(circuit)
display(psi.draw('latex'))


#%% IQFT-3
m = 3 # number of qubits
circuit = QuantumCircuit(m, m) 
iqft = QFT(num_qubits=m,inverse=True).to_gate()
iqft._name = 'IQFT'
circuit.append(iqft, qargs=list(range(m)))
circuit.draw('mpl') 
psi = Statevector(circuit)
display(psi.draw('latex'))

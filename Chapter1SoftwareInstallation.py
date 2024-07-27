# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 07:16:46 2024

@author: Krishnan Suresh; ksuresh@wisc.edu
"""
'''
############### Anaconda installation
conda create -n quantum  
conda activate quantum
conda install pip
conda install spyder
conda install matplotlib
pip install pylatexenc

############### D-Wave installation
conda activate quantum
pip install dwave-system
pip install dwave-neal
pip install dwave-ocean-sdk
pip install pyqubo
dwave config create
'''

import neal
from pyqubo import  Array
import networkx as nx

#%% D-Wave test code
G = nx.Graph() # convenient
G.add_edges_from([(1,2),(1,3),(2,4),(3,4),(3,5),(4,5)])
H = 0
q = Array.create("q",shape = 5,vartype = "BINARY")
for i, j in G.edges:# create H from graph
    H = H + 2*q[i-1]*q[j-1] - q[i-1] - q[j-1] # offset by 1

model = H.compile()
bqm = model.to_bqm()
sampler = neal.SimulatedAnnealingSampler()
results = sampler.sample(bqm)
print(results)

'''
 
############### Qiskit installation
conda activate quantum
pip install qiskit[visualization]
pip install qiskit-ibm-runtime
pip install qiskit-aer

'''

###############
from qiskit_ibm_provider import IBMProvider
IBMProvider.save_account(token='Your API here', overwrite=True)

#%% Qiskit authentication and test
from qiskit_ibm_runtime import QiskitRuntimeService
# QiskitRuntimeService.save_account(channel="ibm_quantum",
# token="Your API here",set_as_default=True, overwrite=True)


#%% Sample Qiskit code to run on a simulator
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
circuit = QuantumCircuit(1, 1) 
circuit.h(0) # apply H to qubit 0
# measure and place result in classical bit
circuit.measure(0, 0) 
circuit.draw('mpl') 

backend = Aer.get_backend('qasm_simulator')
transpiled_circuit = transpile(circuit, backend)
job = backend.run(transpiled_circuit,shots = 1000)
counts = job.result().get_counts(circuit)
print("Counts:\n",counts)

#%% Run on a real IBM quantum machine
from qiskit_ibm_runtime import SamplerV2 as Sampler
if (0): # Change to 1 to run on a real IBM quantum machine
	service = QiskitRuntimeService()
	backend = service.least_busy(operational=True, simulator=False)
	print(backend)
	circuit = QuantumCircuit(1)
	circuit.h(0)
	circuit.measure_all()
	transpiled_circuit = transpile(circuit, backend)
	sampler = Sampler(backend)
	job = sampler.run([transpiled_circuit],shots = 1000)
	print(f"job id: {job.job_id()}")
	result = job.result()
	print(result)
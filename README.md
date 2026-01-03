# Quantum Eraser/Recoherence Experiment on IBM Quantum Platform

Install dependencies:
`pip install qiskit qiskit-aer qiskit-ibm-runtime matplotlib numpy`

Set API key:
`python -c "from qiskit_ibm_runtime import QiskitRuntimeService; QiskitRuntimeService.save_account(channel='ibm_quantum_platform', token='<YOUR_TOKEN_HERE>', overwrite=True)"`

How to run:

```
RUN_HARDWARE=1 \
IBM_BACKEND=ibm_fez \
SHOTS=1024 \
K_VALUES=0,1,2,4,8,16 \
python observer_recoherence_experiment.py
```

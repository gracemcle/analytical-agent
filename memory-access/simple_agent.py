"""
CoT Abstract Reasoning Model for Memory Access

Inputs: 
Device Kernel
Array Size
Block, Grid size
SM Count
Register size
L1/Shared Memory size
L2 size
HBM Size
 |
 V
Memory Analysis Prompt
 |
 V
Synthesis
 |
 V
Runtime Prediction
"""

import os

os.environ['NO_PROXY'] = 'zenith.ftpn.ornl.gov,localhost,127.0.0.1'

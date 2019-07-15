# Low-level Implementations

This part of the project includes programs written in the CUDA API. The considered architectures are:

| GPU                | GeForce930MX               | GTX 1080                           | K80 (2× GK210)                     |
| ------------------ | -------------------------- | ---------------------------------- | ---------------------------------- |
| Purpose in Project | GPU used for local testing | Targeted architecture of the MPIfR | Performance Testing for HPC Taurus |
| Generation         | Maxwell                    | Pascal                             | Kepler                             |
| Compute Capability | 5.0                        | 6.0                                | 3.7                                |
# GPU Polyphase Filterbank for Streaming Radioastronomy

This project includes the CUDA programs for processing data from a telescope. It is developed as part of the master thesis of Nina Herrmann. 

The GPU implementation is separated into two main parts:  **low-level implementations** and **high-level implementations**.
Complementary a CPU program is part of the repository.

## Low-level Implementations
However, for using the programm inside a real system you just need the `CriticalPolyphaseFilterbank.cu/h` files inside [this folder](https://portal.intern.viadee.de/gitlab/viadee/mpifr-kooperation/tree/MPIfR/production/low-level/PFB_radioastronomie/PFB_radioastronomie).

This part of the project includes programs written in the CUDA API. The considered architectures are:

| GPU                | GeForce930MX               | GTX 1080                           | K80 (2× GK210)                     |
| ------------------ | -------------------------- | ---------------------------------- | ---------------------------------- |
| Purpose in Project | GPU used for local testing | Targeted architecture of the MPIfR | Performance Testing for HPC Taurus |
| Generation         | Maxwell                    | Pascal                             | Kepler                             |
| Compute Capability | 5.0                        | 6.0                                | 3.7                                |


## High-level Implementations

The considered frameworks for the high-level implementation will be Musket (Muenster Skeleton Tool). Unfortunately the framework is currently not publicly available. Skeletons provide an additional step of abstraction by including program structures for repeating problems. 


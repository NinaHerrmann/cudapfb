# Low-level Implementations

This part of the project includes programs written in the CUDA API. The considered architectures are:

| GPU                | GeForce930MX               | GTX 1080                           | K80 (2× GK210)                     |
| ------------------ | -------------------------- | ---------------------------------- | ---------------------------------- |
| Purpose in Project | GPU used for local testing | Targeted architecture of the MPIfR | Performance Testing for HPC Taurus |
| Generation         | Maxwell                    | Pascal                             | Kepler                             |
| Compute Capability | 5.0                        | 6.0                                | 3.7                                |

## Limitations
As soon as possible the following issues need to be fixed:
1. The program merely processes number of channels which are a multiple of 32, but 16 would be desirable.
2. Due to the limitation of blocks to 65535 per kernel call, the size of the input is restricted depending on the number of taps and channels. Where the (number of spectra / (56-number of taps)) must be smaller than 65535. This was temporarily fixed with a second kernel call. Now the condition is that (number of spectra / (56-number of taps)) must be smaller than 131070. 

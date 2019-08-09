## High-level Implementations

This part of the project will be started when a functioning low-level implementation has been developed.

The considered framework for the high-level implementation will be Musket (Muenster Skeleton Tool). Unfortunately, the framework is currently not publicly available. Skeletons provide an additional step of abstraction by including program structures for repeating problems. 

To get an idea, a template for the implementation of the polyphase filterbank is now included in the folder. However, it is still work in progress. The FIR Filter is not tested yet, and the FFT is taken from Muesli (the Muenster Skeleton Library).

## Musket Constraints

Some functions are not available in musket therefore numbers which are replaced by the correct C / CUDA functions are included. 

4242 --> __brev(x)

3636 --> -__sinf(temp)

6363 --> __cosf(temp)

8787 --> i ^ (int)


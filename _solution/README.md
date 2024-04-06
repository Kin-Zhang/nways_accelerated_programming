Solution
---


This is not official solution but mine solution... Maybe you can do better than me. I would like to thank Kayran Schmidt for his help. Figure out how to setup cuda one correctly. 

### Result Table

All of these are C/C++ version:

* Running Time is for 500 iterations. with running command `./cfd 64 500`

* All of following are tested on: Desktop setting: i9-12900KF, GPU 3090, with [Dockerfile](../Dockerfile) environment.

|  Approach  | Running Time(s) | Description  |
| ---------- | ----------- | ------------ |
| Raw Serial Baseline        | 7.2941    |   Problem: redundant on swap things|
| ✔️ Improved Serial Baseline      | 4.32531    |   Improve: raw, commit: https://github.com/Kin-Zhang/nways_accelerated_programming/commit/9d4139039d5cc0ece32c69aa76440667338e967a, @(Paul Hoffrogge) is the one who find this |
| ✔️ Compiler Optimization `-O3`      | 1.0917    |  adding `-O3` in CFLAGS on Makefile, compiler optimization  |
| ✔️ ISO stdpar     | 0.0655479    |  v2 std::par inside jacobistep, counting_iterator needed for the fastest speed. complie w gpu  |
| ✔️ OpenACC     | 0.064914    |  check [openacc\jacobi.cpp](openacc\jacobi.cpp). complie w gpu  |
| 🔘 OpenMP     | 0.436107    |  check [openmp\jacobi.cpp](openmp\jacobi.cpp). complie w gpu  |
| ✔️ CUDA unified Memory | 0.063468  |  the easiest way to do cuda  |
| 🔘 CUDA best     |      |     |
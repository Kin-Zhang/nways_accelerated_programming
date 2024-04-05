

N-ways to GPU programming
---
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 
Description: The N-Ways to GPU Programming Bootcamp covers the basics of GPU programming and provides an overview of different methods for porting scientific application to GPUs using NVIDIA® CUDA®, OpenACC, standard languages, OpenMP offloading, and/or CuPy and Numba. Throughout the bootcamp, attendees with learn how to analyze GPU-enabled applications using NVIDIA Nsight™ Systems and participate in hands-on activities to apply these learned skills to real-world problems.

### Bootcamp Schedule

Bootcamp prerequisites: Basic experience with C/C++ or Fortran is needed for the "N-Ways to GPU Programming-C-Fortran" Bootcamp and Python is needed for the "N-Ways to GPU Programming-Python" Bootcamp. No GPU programming experience is required.

0th day – 2 April 2024

- 11:00–12:00	Cluster Dry Run Session

---

1st day – 3 April 2024

- 09:00–09:05	Welcome (Moderator)
- 09:05–09:30	Introduction to GPU Computing (Lecture)
- 09:30–10:00	Introduction to Nsight Systems (Lecture and Read-Only Lab)
- 10:00–11:00	Accelerating Standard C++ and Fortran with GPUs (Lecture and Lab)
- 11:00–11:30	Wrap Up and Q&A

---

2nd day – 4 April 2024

- 09:00–10:30	Directive Based Programming with OpenACC or OpenMP on GPU (Lecture and Lab)
- 10:30–12:30	CUDA C/Fortran Programming (Lecture and Lab)
- 12:30–12:45	Description of Code Challenge
- 12:45–13:00	Wrap Up and Q&A

---

3rd day – 5 April 2024

- 09:00–12:00	Code Challenge
- 09:00–0930	Targeting GPUs from Python [Optional]
- 12:00–12:30	Q&A about Code Challenge
- 12:30–13:00	Project Discussion [Optional]



### Tools and frameworks

- This material originates from the OpenHackathons Github repository. Check out additional materials [here](https://github.com/openhackathons-org).
- [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk)
- [NVIDIA Nsight™ Systems](https://developer.nvidia.com/nsight-systems)



## Instructions

1. Clone the repository:
   ```bash
   git clone 
    ```
2. Build or pull the Docker container:
   ```bash
   docker build -t zhangkin/nways .
   docker pull zhangkin/nways
   ```
3. Run the Docker container:
   ```bash
    docker run --rm -it --runtime nvidia -p 8888:8888 zhangkin/nways
    # if above is not working, try the following
    docker run --gpus all --rm -it -p 8888:8888 zhangkin/nways
    ```
4. Open the Jupyter Notebook in your browser (your local host):
   ```
   http://localhost:8888
   ```
5. Follow the instructions in the Jupyter Notebook to proceed with the bootcamp. Link here is to clone code, visualize as jupyter notebook path.
   - [labs/_start_nways.ipynb](_basic/_start_nways.ipynb).
   - [_challenge/minicfd.ipynb](_challenge/minicfd.ipynb).
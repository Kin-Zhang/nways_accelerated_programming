# Copyright (c) 2020 NVIDIA Corporation.  All rights reserved. 

# HINT 1) comment out the CFLAGS line when you are compiling the parallel code as we no longer need to add the path to the location of nvtx 

# HINT 2) For the CUDA C version, remember to add -arch=native compiler option to use the default architecture on the system you are running.

# HINT 3) For the CUDA C version, make sure the SUFFIXES and CC are correct  

CC=	nvcc
CFLAGS=	-arch=native
LFLAGS=

# System independent definitions

MF=	Makefile

EXE=	cfd

INC= \
	arraymalloc.h \
	boundary.h \
	cfdio.h \
	jacobi.h

# Replace/Add .cu files to the below and make the change for the rest of the locations.
SRC= \
	arraymalloc.cu \
	boundary.cu \
	cfd.cu \
	cfdio.cu \
	jacobi.cu

#
# Make the necessary changes (hint: replace .cu with .cu so the "make clean" does not remove them)
#

.SUFFIXES:
.SUFFIXES: .cu .o

OBJ=	$(SRC:.cu=.o)

.cu.o:
	$(CC) $(CFLAGS) -c $<

all:	$(EXE)

$(OBJ):	$(INC)

$(EXE):	$(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LFLAGS)

$(OBJ):	$(MF)

tar:
	tar cvf cfd.tar $(MF) $(INC) $(SRC)

clean:
	rm -f $(OBJ) $(EXE) velocity.dat colourmap.dat cfd.plt core

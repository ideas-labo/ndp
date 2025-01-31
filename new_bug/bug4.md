## **Anomalous Performance of `-O0`**

In GCC 12.0, we observed that for certain test cases, the `-O0` optimization level outperformed `-O1`, but this phenomenon was not observed in GCC 9.2. Due to limited resources, we only reported anomalies that consistently appeared in repeated tests. System noise cannot be completely ruled out. The Docker image used for testing is available [here](https://hub.docker.com/r/anonymicse2021/gcc_inputs). Test data are as follows:

### **Test Details**

```bash

inputlocation="linear-algebra/kernels/trisolv"
inputname="trisolv"

# GCC 12.0
gcc -O0 -fprefetch-loop-arrays -ffloat-store -fno-asm -I ./inputs/utilities -I ./inputs/linear-algebra/kernels/trisolv ./inputs/utilities/polybench.c ./inputs/linear-algebra/kernels/trisolv/trisolv.c -DPOLYBENCH_TIME -o ./test
time ./test
real    0m0.183s
user    0m0.114s
sys     0m0.069s

gcc -O1 -fprefetch-loop-arrays -ffloat-store -fno-asm -I ./inputs/utilities -I ./inputs/linear-algebra/kernels/trisolv ./inputs/utilities/polybench.c ./inputs/linear-algebra/kernels/trisolv/trisolv.c -DPOLYBENCH_TIME -o ./test
time ./test
real    0m0.218s
user    0m0.132s
sys     0m0.086s

# GCC 9.2
gcc -O0  -fprefetch-loop-arrays -ffloat-store -fno-asm  -I ./inputs/utilities -I ./inputs/$inputlocation ./inputs/utilities/polybench.c ./inputs/$inputlocation/$inputname.c -DPOLYBENCH_TIME -o ./test

time ./test
real    0m0.379s
user    0m0.316s
sys     0m0.062s

gcc -O1  -fprefetch-loop-arrays -ffloat-store -fno-asm  -I ./inputs/utilities -I ./inputs/$inputlocation ./inputs/utilities/polybench.c ./inputs/$inputlocation/$inputname.c -DPOLYBENCH_TIME -o ./test

time ./test
real    0m0.215s
user    0m0.131s
sys     0m0.085s
 

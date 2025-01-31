## **`-O1` Compilation Slower than `-O2`**

During testing with GCC 9.2, we observed that for certain test cases, the compilation speed with `-O1` was slower than with `-O2`. This phenomenon was not observed in GCC 12.0. Due to limited resources, we only documented anomalies that consistently appeared across multiple tests. System noise cannot be entirely ruled out. The Docker image used for testing is available [here](https://hub.docker.com/r/anonymicse2021/gcc_inputs). The test results are as follows:

### **Test Details**

```bash

inputlocation="linear-algebra/kernels/gemm"
inputname="gemm"

# GCC 9.2
time gcc -O2 -floop-interchange -I ./inputs/utilities -I ./inputs/linear-algebra/kernels/gemm ./inputs/utilities/polybench.c ./inputs/linear-algebra/kernels/gemm/gemm.c -DPOLYBENCH_TIME -o ./test
real    0m0.158s
user    0m0.109s
sys     0m0.048s

time gcc -O1 -floop-interchange -I ./inputs/utilities -I ./inputs/linear-algebra/kernels/gemm ./inputs/utilities/polybench.c ./inputs/linear-algebra/kernels/gemm/gemm.c -DPOLYBENCH_TIME -o ./test
real    0m0.174s
user    0m0.124s
sys     0m0.048s

# GCC 12.0
time gcc -O2 -floop-interchange     -I ./inputs/utilities -I ./inputs/$inputlocation ./inputs/utilities/polybench.c ./inputs/$inputlocation/$inputname.c -DPOLYBENCH_TIME -o ./test

real    0m0.170s
user    0m0.125s
sys     0m0.044s

time gcc -O1 -floop-interchange     -I ./inputs/utilities -I ./inputs/$inputlocation ./inputs/utilities/polybench.c ./inputs/$inputlocation/$inputname.c -DPOLYBENCH_TIME -o ./test

real    0m0.124s
user    0m0.086s
sys     0m0.038s

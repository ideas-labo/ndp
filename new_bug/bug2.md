## **Performance Impact of `-ffloat-store`**

While testing GCC 9.2 and GCC 12.0, we observed some anomalies. In certain test cases, enabling `-ffloat-store` caused a significant increase in execution time, nearly doubling it in some instances. Meanwhile, enabling `-Og` optimizations was almost ineffective and sometimes even degraded performance. Due to limited resources, we reported only the consistently observed anomalies and could not eliminate the potential influence of random factors. The Docker image used for testing is available [here](https://hub.docker.com/r/anonymicse2021/gcc_inputs). The detailed test data are as follows:

### **Test Details**

```bash

inputlocation="linear-algebra/kernels/doitgen"
inputname="doitgen"

gcc -g -ffloat-store -I ./inputs/utilities -I ./inputs/linear-algebra/kernels/doitgen ./inputs/utilities/polybench.c ./inputs/linear-algebra/kernels/doitgen/doitgen.c -DPOLYBENCH_TIME -o ./test
time ./test
real    0m1.765s
user    0m1.709s
sys     0m0.055s

# GCC with -ffloat-store (another run)
gcc -ffloat-store -I ./inputs/utilities -I ./inputs/linear-algebra/kernels/doitgen ./inputs/utilities/polybench.c ./inputs/linear-algebra/kernels/doitgen/doitgen.c -DPOLYBENCH_TIME -o ./test
time ./test
real    0m1.700s
user    0m1.658s
sys     0m0.041s

# GCC without -ffloat-store
gcc -I ./inputs/utilities -I ./inputs/linear-algebra/kernels/doitgen ./inputs/utilities/polybench.c ./inputs/linear-algebra/kernels/doitgen/doitgen.c -DPOLYBENCH_TIME -o ./test
time ./test
real    0m0.920s
user    0m0.878s
sys     0m0.041s
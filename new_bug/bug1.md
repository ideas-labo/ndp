## **Performance Difference of `-fno-asm` Across GCC Versions**

Recently, while testing various GCC versions using Polybench 3.1, we observed an interesting behavior: enabling `-fno-asm` in GCC 12.0 improves performance, whereas disabling it in GCC 9.2 results in better performance. However, this phenomenon was not observed in the Clang versions we tested. Due to limited resources, we conducted only a few tests and reported the performance anomalies that occurred consistently across these runs. System noise cannot be completely ruled out. The Docker image used for testing is publicly available [here](https://hub.docker.com/r/anonymicse2021/gcc_inputs). Detailed test results are as follows:

### **Test Details**

```bash

inputlocation="linear-algebra/solvers/lu"
inputname="lu"

# GCC 12.0 - With -fno-asm
gcc -O2 -fprefetch-loop-arrays -fno-asm -I ./inputs/utilities -I ./inputs/linear-algebra/solvers/lu ./inputs/utilities/polybench.c ./inputs/linear-algebra/solvers/lu/lu.c -DPOLYBENCH_TIME -o ./test
time ./test
real    0m0.236s
user    0m0.207s
sys     0m0.030s

# GCC 12.0 - Without -fno-asm
gcc -O2 -fprefetch-loop-arrays -I ./inputs/utilities -I ./inputs/linear-algebra/solvers/lu ./inputs/utilities/polybench.c ./inputs/linear-algebra/solvers/lu/lu.c -DPOLYBENCH_TIME -o ./test
time ./test
real    0m0.268s
user    0m0.235s
sys     0m0.033s

# GCC 9.2 - With -fno-asm
gcc -O2 -fprefetch-loop-arrays -fno-asm -I ./inputs/utilities -I ./inputs/linear-algebra/solvers/lu ./inputs/utilities/polybench.c ./inputs/linear-algebra/solvers/lu/lu.c -DPOLYBENCH_TIME -o ./test
time ./test
real    0m0.265s
user    0m0.233s
sys     0m0.033s

# GCC 9.2 - Without -fno-asm
gcc -O2 -fprefetch-loop-arrays -I ./inputs/utilities -I ./inputs/linear-algebra/solvers/lu ./inputs/utilities/polybench.c ./inputs/linear-algebra/solvers/lu/lu.c -DPOLYBENCH_TIME -o ./test
time ./test
real    0m0.247s
user    0m0.216s
sys     0m0.030s

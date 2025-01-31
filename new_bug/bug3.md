## **Unusual Compilation Times**

During performance testing of GCC 9.2 and GCC 12.0, we noticed that for some test cases, the compilation time with `-O1` was very close to that of `-Ofast` or `-O2`, and in some instances even slower. This behavior was also observed in certain Clang versions. Due to limited resources, we reported only anomalies that consistently appeared in repeated tests. System noise cannot be entirely ruled out. The Docker image used for testing is available [here](https://hub.docker.com/r/anonymicse2021/gcc_inputs). The test data are as follows:

### **Test Details**

```bash

inputlocation="linear-algebra/kernels/bicg"
inputname="bicg"

time gcc -Ofast -fprefetch-loop-arrays -ffloat-store -I ./inputs/utilities -I ./inputs/linear-algebra/kernels/bicg ./inputs/utilities/polybench.c ./inputs/linear-algebra/kernels/bicg/bicg.c -DPOLYBENCH_TIME -o ./test
real    0m0.171s
user    0m0.127s
sys     0m0.044s

time gcc -Ofast -fprefetch-loop-arrays -ffloat-store -I ./inputs/utilities -I ./inputs/linear-algebra/kernels/bicg ./inputs/utilities/polybench.c ./inputs/linear-algebra/kernels/bicg/bicg.c -DPOLYBENCH_TIME -o ./test
real    0m0.154s
user    0m0.113s
sys     0m0.041s

inputlocation="linear-algebra/kernels/2mm"
inputname="2mm"

time gcc -O0 -fprefetch-loop-arrays -ffloat-store -fno-asm -I ./inputs/utilities -I ./inputs/linear-algebra/kernels/2mm ./inputs/utilities/polybench.c ./inputs/linear-algebra/kernels/2mm/2mm.c -DPOLYBENCH_TIME -o ./test
real    0m0.116s
user    0m0.075s
sys     0m0.040s

time gcc -O1 -fprefetch-loop-arrays -ffloat-store -fno-asm -I ./inputs/utilities -I ./inputs/linear-algebra/kernels/2mm ./inputs/utilities/polybench.c ./inputs/linear-algebra/kernels/2mm/2mm.c -DPOLYBENCH_TIME -o ./test
real    0m0.232s
user    0m0.187s
sys     0m0.045s

time gcc -O2  -fprefetch-loop-arrays -ffloat-store -fno-asm  -I ./inputs/utilities -I ./inputs/$inputlocation ./inputs/utilities/polybench.c ./inputs/$inputlocation/$inputname.c -DPOLYBENCH_TIME -o ./test

real    0m0.220s
user    0m0.171s
sys     0m0.048s

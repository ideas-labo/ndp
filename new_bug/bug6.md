## **`-O0` Compilation Slower than `-Og`**

In GCC 12.0, we observed that for certain test cases, the compilation speed with `-O0` was slower than that with `-Og`. This phenomenon was not observed in GCC 9.2. Due to limited resources, we only reported anomalies that consistently appeared across repeated tests. System noise cannot be completely ruled out. The Docker image used for testing is available [here](https://hub.docker.com/r/anonymicse2021/gcc_inputs). Test data are as follows:

### **Test Details**

```bash
inputlocation="linear-algebra/kernels/doitgen"
inputname="doitgen"

# GCC 12.0
time gcc -Og -ffloat-store -I ./inputs/utilities -I ./inputs/linear-algebra/kernels/doitgen ./inputs/utilities/polybench.c ./inputs/linear-algebra/kernels/doitgen/doitgen.c -DPOLYBENCH_TIME -o ./test
real    0m0.112s
user    0m0.080s
sys     0m0.032s

time gcc -O0 -ffloat-store -I ./inputs/utilities -I ./inputs/linear-algebra/kernels/doitgen ./inputs/utilities/polybench.c ./inputs/linear-algebra/kernels/doitgen/doitgen.c -DPOLYBENCH_TIME -o ./test
real    0m0.131s
user    0m0.086s
sys     0m0.044s

# GCC 9.2
time gcc -Og   -ffloat-store   -I ./inputs/utilities -I ./inputs/$inputlocation ./inputs/utilities/polybench.c ./inputs/$inputlocation/$inputname.c -DPOLYBENCH_TIME -o ./test

real    0m0.136s
user    0m0.096s
sys     0m0.041s

time gcc -O0   -ffloat-store   -I ./inputs/utilities -I ./inputs/$inputlocation ./inputs/utilities/polybench.c ./inputs/$inputlocation/$inputname.c -DPOLYBENCH_TIME -o ./test

real    0m0.118s
user    0m0.075s
sys     0m0.043s

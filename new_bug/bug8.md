## **`-ffloat-store` Increases Executable Size**

While testing GCC 9.2 and GCC 12.0, we observed that enabling `-ffloat-store` significantly increased the size of the generated executables for certain test cases. However, this behavior did not occur when combined with `-O2` or higher optimization levels. Due to limited resources, we reported only consistently observed anomalies. The Docker image used for testing is available [here](https://hub.docker.com/r/anonymicse2021/gcc_inputs). The test data are as follows:

### **Test Details**

```bash
inputlocation="stencils/adi"
inputname="adi"

gcc -ffloat-store   -I ./inputs/utilities -I ./inputs/$inputlocation ./inputs/utilities/polybench.c ./inputs/$inputlocation/$inputname.c -DPOLYBENCH_TIME -o ./test

stat -c%s test
21848

gcc -I ./inputs/utilities -I ./inputs/$inputlocation ./inputs/utilities/polybench.c ./inputs/$inputlocation/$inputname.c -DPOLYBENCH_TIME -o ./test

stat -c%s test
17752


gcc -Ofast -ffloat-store   -I ./inputs/utilities -I ./inputs/$inputlocation ./inputs/utilities/polybench.c ./inputs/$inputlocation/$inputname.c -DPOLYBENCH_TIME -o ./test

stat -c%s test
23208

gcc -Ofast -I ./inputs/utilities -I ./inputs/$inputlocation ./inputs/utilities/polybench.c ./inputs/$inputlocation/$inputname.c -DPOLYBENCH_TIME -o ./test

stat -c%s test
19112


gcc -O3 -ffloat-store   -I ./inputs/utilities -I ./inputs/$inputlocation ./inputs/utilities/polybench.c ./inputs/$inputlocation/$inputname.c -DPOLYBENCH_TIME -o ./test

stat -c%s test
21696

gcc -O3 -I ./inputs/utilities -I ./inputs/$inputlocation ./inputs/utilities/polybench.c ./inputs/$inputlocation/$inputname.c -DPOLYBENCH_TIME -o ./test

stat -c%s test
17600

gcc -O1 -ffloat-store   -I ./inputs/utilities -I ./inputs/$inputlocation ./inputs/utilities/polybench.c ./inputs/$inputlocation/$inputname.c -DPOLYBENCH_TIME -o ./test

stat -c%s test
17584

gcc -O1 -I ./inputs/utilities -I ./inputs/$inputlocation ./inputs/utilities/polybench.c ./inputs/$inputlocation/$inputname.c -DPOLYBENCH_TIME -o ./test

stat -c%s test
17584

gcc -O2 -ffloat-store   -I ./inputs/utilities -I ./inputs/$inputlocation ./inputs/utilities/polybench.c ./inputs/$inputlocation/$inputname.c -DPOLYBENCH_TIME -o ./test

stat -c%s test
17552

gcc -O2 -I ./inputs/utilities -I ./inputs/$inputlocation ./inputs/utilities/polybench.c ./inputs/$inputlocation/$inputname.c -DPOLYBENCH_TIME -o ./test

stat -c%s test
17552

gcc -Os -ffloat-store   -I ./inputs/utilities -I ./inputs/$inputlocation ./inputs/utilities/polybench.c ./inputs/$inputlocation/$inputname.c -DPOLYBENCH_TIME -o ./test

stat -c%s test
17584

gcc -Os -I ./inputs/utilities -I ./inputs/$inputlocation ./inputs/utilities/polybench.c ./inputs/$inputlocation/$inputname.c -DPOLYBENCH_TIME -o ./test

stat -c%s test
17584

gcc -Og -ffloat-store   -I ./inputs/utilities -I ./inputs/$inputlocation ./inputs/utilities/polybench.c ./inputs/$inputlocation/$inputname.c -DPOLYBENCH_TIME -o ./test

stat -c%s test
17720

gcc -Og  -I ./inputs/utilities -I ./inputs/$inputlocation ./inputs/utilities/polybench.c ./inputs/$inputlocation/$inputname.c -DPOLYBENCH_TIME -o ./test

stat -c%s test
17720

# This is a test case that does not increase the size
inputlocation="linear-algebra/kernels/gesummv"
inputname="gesummv"

gcc -Ofast -ffloat-store   -I ./inputs/utilities -I ./inputs/$inputlocation ./inputs/utilities/polybench.c ./inputs/$inputlocation/$inputname.c -DPOLYBENCH_TIME -o ./test

stat -c%s test
19112

gcc -Ofast -I ./inputs/utilities -I ./inputs/$inputlocation ./inputs/utilities/polybench.c ./inputs/$inputlocation/$inputname.c -DPOLYBENCH_TIME -o ./test

stat -c%s test
19112

gcc -ffloat-store   -I ./inputs/utilities -I ./inputs/$inputlocation ./inputs/utilities/polybench.c ./inputs/$inputlocation/$inputname.c -DPOLYBENCH_TIME -o ./test

stat -c%s test
17760

gcc -I ./inputs/utilities -I ./inputs/$inputlocation ./inputs/utilities/polybench.c ./inputs/$inputlocation/$inputname.c -DPOLYBENCH_TIME -o ./test

stat -c%s test
17760
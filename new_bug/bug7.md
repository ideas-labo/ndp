# **Certain Test Cases Cannot Be Optimized**

During testing with GCC 9.2 and GCC 12.0, we found that certain matrix computation test cases showed little to no performance improvement under most optimization levels and, in some cases, even degraded. Similar behavior was observed in some Clang versions as well. Due to limited resources, we reported only consistently observed anomalies and could not eliminate the potential influence of random factors. The Docker image used for testing is available [here](https://hub.docker.com/r/anonymicse2021/gcc_inputs). Detailed test results are as follows:

### **Test Details**

```bash
inputlocation="linear-algebra/kernels/symm"
inputname="symm"

gcc -O0 -I ./inputs/utilities -I ./inputs/$inputlocation ./inputs/utilities/polybench.c ./inputs/$inputlocation/$inputname.c -DPOLYBENCH_TIME -o ./test

time ./test
real	0m6.684s
user	0m6.633s
sys	0m0.047s

gcc -O1 -I ./inputs/utilities -I ./inputs/$inputlocation ./inputs/utilities/polybench.c ./inputs/$inputlocation/$inputname.c -DPOLYBENCH_TIME -o ./test

time ./test
real	0m7.210s
user	0m7.161s
sys	0m0.046s

gcc -O2 -I ./inputs/utilities -I ./inputs/$inputlocation ./inputs/utilities/polybench.c ./inputs/$inputlocation/$inputname.c -DPOLYBENCH_TIME -o ./test

time ./test
real	0m7.343s
user	0m7.301s
sys	0m0.041s

gcc -O3 -I ./inputs/utilities -I ./inputs/$inputlocation ./inputs/utilities/polybench.c ./inputs/$inputlocation/$inputname.c -DPOLYBENCH_TIME -o ./test

time ./test
real	0m7.310s
user	0m7.267s
sys	0m0.037s

gcc -Ofast -I ./inputs/utilities -I ./inputs/$inputlocation ./inputs/utilities/polybench.c ./inputs/$inputlocation/$inputname.c -DPOLYBENCH_TIME -o ./test

time ./test
real	0m7.074s
user	0m7.021s
sys	0m0.050s

gcc -Og -I ./inputs/utilities -I ./inputs/$inputlocation ./inputs/utilities/polybench.c ./inputs/$inputlocation/$inputname.c -DPOLYBENCH_TIME -o ./test

time ./test
real	0m7.180s
user	0m7.138s
sys	0m0.033s

nvcc -ccbin g++ -I ./ -m64  -gencode arch=compute_75,code=compute_75 -o color_debayer_test4 color_debayer_test4.cpp -I/usr/local/cuda-11.4/include -L/usr/local/cuda-11.4/lib64 \
    -lcuda -lnppicc
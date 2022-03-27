#include <thrust/device_vector.h>
#include <iostream>
#include <stdlib.h>
#include "cuda_utils.cuh"
#include <getopt.h>

void parse_opts(int argc, char **argv, 
    int *blksize, // block size 
     int *n)      // problem size
{

    int opt; 
     
    while ((opt = getopt(argc, argv, "pb:")) != -1) {

        switch(opt) {
            case 'b' : {
                *blksize = atoi(optarg);
                break;
            } case 'p' : {
                *n = atoi(optarg);
                break;
            } default :
                break;
        }
    }
}

void basicThrustTest(int n){

    // Given X,Y   compute  X = (X^2 + X - Y) % Y
    double start = clock();
    std::cout << "Thrust slow compute test\n" << std::endl;

    thrust::device_vector<int> X(n);
    thrust::device_vector<int> Y(n);
    thrust::device_vector<int> Z(n);
    
    thrust::sequence(X.begin(), X.end());
    thrust::fill(Y.begin(), Y.end(), 22);
    
    // Z = X*X
    thrust::transform(X.begin(), X.end(), 
        X.begin(), 
        Z.begin(),  
        thrust::multiplies<int>()
    );

    // Z = Z + X
    thrust::transform(Z.begin(), Z.end(), 
        X.begin(), 
        Z.begin(),  
        thrust::plus<int>()
    );
    // Z = Z - Y
    thrust::transform(Z.begin(), Z.end(), 
        Y.begin(), 
        Z.begin(),  
        thrust::minus<int>()
    );
    // Z = Z % Y
    thrust::transform(Z.begin(), Z.end(), 
        Y.begin(), 
        Z.begin(),  
        thrust::modulus<int>()
    );

   
    int sel = Z.size() -1; 
    std::cout << "Z[" << sel << "] = " << Z[sel] << std::endl; 
 
    double diff = clock() - start;
    std::cout << "basic test Time spent " << diff << std::endl;
    
    return;
}

struct fast_functor {
    
    __host__ __device__
        float operator()(const int &x, const int &y) const{
            return (x*x + x - y ) % y;
        }
};

void compoundThrustTest(int n){

    // Given X,Y   compute   (X^2 + X - Y) % Y
    double start = clock();
    std::cout << "Thrust fast compute test\n" << std::endl;

    thrust::device_vector<int> X(n);
    thrust::device_vector<int> Y(n);
    thrust::device_vector<int> Z(n);
    
    thrust::sequence(X.begin(), X.end());
    thrust::fill(Y.begin(), Y.end(), 22);
    thrust::transform(X.begin(), X.end(),
        Y.begin(),
        Z.begin(),
        fast_functor()
    );
    
    int sel = Z.size() -1; 
    std::cout << "Z[" << sel << "] = " << Z[sel] << std::endl; 
    
    double diff = clock() - start;
    std::cout << "fast test Time spent " << diff << std::endl;

}

int main(int argc, char **argv){

    int blocksize, n;
    parse_opts(argc, argv, &blocksize, &n);
    std::cout << "testing" << std::endl;

    basicThrustTest(n);
    compoundThrustTest(n);

}

#include <thrust/device_vector.h>
#include <iostream>
#include <stdlib.h>
#include "cuda_utils.cuh"
#include <argp.h>


static int problem_size = 0;
static int blocksize;

static int parse_opt(int key, char *arg, struct argp_state *state)
{

    if( arg == NULL) {
        return 0;
    }

    switch (key) {
        
        case 'p': {
            problem_size = atoi(arg);
            printf("n = %d\n", problem_size);
            break; 
        } case 'b' : {
            blocksize = atoi(arg);
            printf("blocksize = %d\n", blocksize);
            break;
        }

    }
    return 0;
}

struct argp_option options[] = 
{
    {"size", 'p', "NUM", OPTION_ARG_OPTIONAL, "Problem size to work on"},
    {"bsize", 'b', "NUM", OPTION_ARG_OPTIONAL, "Block size"},
    { 0 }
};

void basicThrustTest(int n){

    // Given X,Y   compute  X = (X^2 + X - Y) % Y
    thrust::host_vector<int> H(n);

    thrust::generate(H.begin(), H.end(), rand); // generate vectors on host
    thrust::device_vector<int> X = H;

    thrust::generate(H.begin(), H.end(), rand);
    thrust::device_vector<int> Y = H;

    thrust::device_vector<int> Z(n);
    
    
    int sel = rand() % n;  // display vectors selected valued
    std::cout << "X[" << sel << "] = " << X[sel] << std::endl;
    std::cout << "Y[" << sel << "] = " << Y[sel] << std::endl;
    
    double start = clock(); // start timing after to compare a compact functor verse
    std::cout << "Thrust slow compute test\n" << std::endl; // a non compact series
 
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

    std::cout << "Z[" << sel << "] = " << Z[sel] << std::endl; 
 
    double diff = clock() - start;
    std::cout << "basic test Time spent " << diff << std::endl;
    
    return;
}

// by streamlining the kernel of the functor we execute faster
struct fast_functor {
    __host__ __device__
        float operator()(const int &x, const int &y) const{
            return (x*x + x - y ) % y;
        }
};

void compoundThrustTest(int n){

    // Given X,Y   compute   (X^2 + X - Y) % Y

    thrust::host_vector<int> H(n);

    thrust::generate(H.begin(), H.end(), rand); // generate with rand on host side
    thrust::device_vector<int> X = H;

    thrust::generate(H.begin(), H.end(), rand);
    thrust::device_vector<int> Y = H;

    thrust::device_vector<int> Z(n);
    

    int sel = rand() % n;  // display selcted values
    std::cout << "X[" << sel << "] = " << X[sel] << std::endl;
    std::cout << "Y[" << sel << "] = " << Y[sel] << std::endl;
    
    double start = clock(); // start timing after generation
    std::cout << "Thrust fast compute test\n" << std::endl;

    thrust::transform(X.begin(), X.end(),
        Y.begin(),
        Z.begin(),
        fast_functor()
    );
    
    std::cout << "Z[" << sel << "] = " << Z[sel] << std::endl; 
    
    double diff = clock() - start;
    std::cout << "fast test Time spent " << diff << std::endl;

}

int main(int argc, char **argv){

    struct argp argp = {options, parse_opt, 0, 0};
    argp_parse(&argp, argc, argv, 0, 0, 0);
    srand(time(NULL));
    int n = problem_size;

    basicThrustTest(n);
    compoundThrustTest(n);

    return 0;
}

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

void hello(int n) {

    thrust::host_vector<int> H(n);
    H[0] = 1;
    H[4] = 4;
    std::cout << "H[4]=" << H[4] << " overall H has len=" << H.size() << std::endl; 

    thrust::device_vector<int> D = H;
    D[4] = 8;
    std::cout << "D[0]=" << D[0] << " overall D has len=" << D.size() << std::endl; 
    std::cout << "D[4]=" << D[4] << " overall D has len=" << D.size() << std::endl; 

    thrust::device_vector<int> C(n);
    C[4] = 9;
    std::cout << "C[4]=" << C[4] << " overall C has len=" << C.size() << std::endl; 

    thrust::generate(thrust::host, H.begin(), H.end(), rand);
    std::cout << "H[4]=" << H[4] << " overall H has len=" << H.size() << std::endl; 
    
    thrust::generate(thrust::device, D.begin(), D.end(), rand);
    std::cout << "D[0]=" << D[0] << " overall D has len=" << D.size() << std::endl; 
    std::cout << "D[4]=" << D[4] << " overall D has len=" << D.size() << std::endl; 

    return;

}

void basicThrustTest(int n){

    // Given X,Y   compute  X = (X^2 + X - Y) % Y
    double start = clock();
    std::cout << "Thrust slow compute test\n" << std::endl;

    thrust::host_vector<int> H(n);

    thrust::generate(H.begin(), H.end(), rand);
    thrust::device_vector<int> X = H;

    thrust::generate(H.begin(), H.end(), rand);
    thrust::device_vector<int> Y = H;

    thrust::device_vector<int> Z(n);
    
    int sel = rand() % n; 
    thrust::generate(X.begin(), X.end(), rand);
    thrust::generate(Y.begin(), Y.end(), rand);
   
    std::cout << "X[" << sel << "] = " << X[sel] << std::endl;
    std::cout << "Y[" << sel << "] = " << Y[sel] << std::endl;
 
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
    
    thrust::generate(X.begin(), X.end(), rand);
    thrust::generate(Y.begin(), Y.end(), rand);

    thrust::transform(X.begin(), X.end(),
        Y.begin(),
        Z.begin(),
        fast_functor()
    );
    
    int sel = rand() % n; 
    std::cout << "Z[" << sel << "] = " << Z[sel] << std::endl; 
    
    double diff = clock() - start;
    std::cout << "fast test Time spent " << diff << std::endl;

}

int main(int argc, char **argv){

    struct argp argp = {options, parse_opt, 0, 0};
    argp_parse(&argp, argc, argv, 0, 0, 0);
    srand(time(NULL));
    int n = problem_size;

    hello(n);
    basicThrustTest(n);
    compoundThrustTest(n);

    return 0;
}

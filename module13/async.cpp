#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "info.hpp"
#include <argp.h>
#include <cstring>

inline void
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

// argp funcs start

typedef struct {
    int n;
    int eq;
    int mode;
    int order;
}pargs;

static int parse_opt(int key, char *arg, struct argp_state *state)
{

    pargs *args = (pargs*) state->input;
    switch(key){
    
        case 'n': {
           args->n = atoi(arg); 
            break;
        } case 'q': {
            args->eq = atoi(arg);
            break;
        } case 'c': {
            args->mode = atoi(arg);
            break;
        } case 'o': {
            args->order = atoi(arg);
        }

    }
    return 0;
}

static struct argp_option options[] = 
{
    {"size", 'n', "size", 0, "Problem size"},
    {"queued", 'q', "queued", 0, "Number of events/kernels to queue up"},
    {"callback", 'c', "callback", 0, "Mode for call backs"},
    {"order", 'o', 0, 0, "Order of operations"},
    {0}
};

static struct argp argp = {options, parse_opt, 0, 0};

// argp funcs end


void opencl_bootstrap(cl_context *context, cl_command_queue *queue, cl_program *program, const char *src_name)
{

    cl_int errNum;
    cl_uint numPlatforms;
    cl_uint numDevices;
    int platform = 0;

    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr(
        (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS),
        "clGetPlatformIDs");
    
    cl_platform_id* platformIDs = (cl_platform_id *) alloca(sizeof(cl_platform_id) * numPlatforms);
    
    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr(
       (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS),
       "clGetPlatformIDs");

    std::ifstream srcFile(src_name);
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading simple.cl");

    std::string srcProg(
        std::istreambuf_iterator<char>(srcFile),
        (std::istreambuf_iterator<char>()));

    const char * src = srcProg.c_str();
    size_t length = srcProg.length();

	cl_device_id * deviceIDs = NULL;

    errNum = clGetDeviceIDs(
        platformIDs[platform],
        CL_DEVICE_TYPE_ALL,
        0,
        NULL,
        &numDevices);
    if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
    {
        checkErr(errNum, "clGetDeviceIDs");
    } else if (numDevices > 0) {
		deviceIDs = (cl_device_id*) alloca(sizeof(cl_device_id) * numDevices);
		 errNum = clGetDeviceIDs(
                platformIDs[platform], // BUG??
                CL_DEVICE_TYPE_GPU,
                numDevices,
                &deviceIDs[0],
                NULL);
            checkErr(errNum, "clGetDeviceIDs");
	}
    
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformIDs[platform],
        0
    };


    *context = clCreateContext(
        contextProperties,
        numDevices,
        deviceIDs,
        NULL,
        NULL,
        &errNum);
    checkErr(errNum, "clCreateContext");
    
    *program = clCreateProgramWithSource(
        *context,
        1,
        &src,
        &length,
        &errNum);
    checkErr(errNum, "clCreateProgramWithSource");

	errNum = clBuildProgram(
        *program,
        numDevices,
        deviceIDs,
        NULL,
        NULL,
        NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(
            *program,
            deviceIDs[0],
            CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog),
            buildLog,
            NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        checkErr(errNum, "clBuildProgram");
    }

    // Pick the first device and create command queue.
    *queue = clCreateCommandQueue(
        *context,
        deviceIDs[0],
        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, // everything requires events
        &errNum);
    checkErr(errNum, "clCreateCommandQueue");
}

void init_args(pargs* args)
{
	args->n = 1024;
	args->eq = 1;
	args->mode = 0;
	args->order = 0;
}

typedef struct kern_cb_data{
    int kIdx;
    int kBufSize;
    cl_mem kBuf;
    cl_command_queue queue;
}kern_cb_data_t;

void CL_CALLBACK event_cb(cl_event event, cl_int status, void* data){
    printf("CL callback!\n");
    kern_cb_data_t *kparam = (kern_cb_data_t*)data;
    printf("kern %d callback status: %d\n", kparam->kIdx, status);
    int* out = new int[kparam->kBufSize];
    cl_int errNum = clEnqueueReadBuffer(kparam->queue, kparam->kBuf, CL_TRUE, 0, kparam->kBufSize,
                        (void*)out, 0, NULL, NULL);
    checkErr(errNum, "Error?");

    printf("first 3 values %d %d %d \n", out[0], out[1], out[2]);
    delete out;
}
	
int get_blocker(int kIdx, cl_event *events, cl_event *blocker, pargs *args)
{
    printf("Get the wait list for %d\n", kIdx); 
    if(args->mode == 0 || kIdx == 0){ // no order
        return 0;
    } else if(args->mode == 1){ // in order
        *blocker = events[kIdx];
        return 1;
    } else if(args->mode == 2){ // even odd
        if(kIdx == 1){
            return 0;
        }
        *blocker = events[kIdx-2];
        return 1;
    }
    return 0; // no waiting if fail

}


int kernIn[5] = {0,1,2,3,4};

void launchKernelTree(cl_context *context, cl_command_queue *queue, cl_program *program, cl_event *events, int kIdx, pargs *args)
{
    cl_int errNum;
    cl_kernel kern = clCreateKernel(
        *program,
        "cube",
        &errNum
    );
    checkErr(errNum, "clCreateKernel");

    kern_cb_data_t* cdata = new kern_cb_data_t;
    cdata->kIdx = kIdx;
    cdata->queue = *queue;
    cdata->kBuf = clCreateBuffer(
        *context,
        CL_MEM_READ_WRITE,
        sizeof(int) * 5,
        NULL,
        &errNum
    );
    checkErr(errNum, "clCreateBuffer");
    cdata->kBufSize = 5;

    errNum = clSetKernelArg(kern, 0, sizeof(cl_mem), (void *)&(cdata->kBuf));
    checkErr(errNum, "clSetKernelArg");
    
    errNum = clEnqueueWriteBuffer(
        cdata->queue,
        cdata->kBuf,
        CL_TRUE,
        0,
        sizeof(int) * 5,
        kernIn,
        0, NULL, NULL);
    checkErr(errNum, "clWriteBuff");
    
    cl_event blocker = NULL;
    int numBlocker = get_blocker(kIdx, events, &blocker, args);
    printf("Kernel %d has %d blockers, %llx\n", kIdx, numBlocker, (long long)blocker);
    

    size_t gWI = 5;
    if(numBlocker == 0){

        errNum = clEnqueueNDRangeKernel(cdata->queue, kern, 1, NULL,
            (const size_t*)&gWI, (const size_t*)NULL, 0, NULL, &events[kIdx]);
    } else {
        errNum = clEnqueueNDRangeKernel(cdata->queue, kern, 1, NULL,
            (const size_t*)&gWI, (const size_t*)NULL, numBlocker, &blocker, &events[kIdx]);

    }
    checkErr(errNum, "Error with kernel enqueu");

    errNum = clSetEventCallback(events[kIdx], CL_COMPLETE, event_cb, cdata);
    checkErr(errNum, "set call back");
    printf("Set callback success!");
    
    
}

int main(int argc, char** argv){
    
    // input
    pargs args;
	init_args(&args);    
	argp_parse(&argp, argc, argv, 0, 0, &args);

    // opencl context
    cl_context context = NULL;
    cl_command_queue queue;
    cl_program program;
    
    opencl_bootstrap(&context, &queue, &program, "simple.cl"); 
	cl_event* events = new cl_event[args.eq+1];
    
	for(int i = 0; i < args.eq; i++)
	{
        printf("Launching Kernel %d\n", i);
		launchKernelTree(&context, &queue, &program, events, i, &args);
	}

    return 0;

}

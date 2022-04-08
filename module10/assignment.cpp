// Credit to opencude guide project for its base

#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

///
//  Constants
//
const int ARRAY_SIZE = 1000;

///
//  Create an OpenCL context on the first available platform using
//  either a GPU or CPU depending on what is available.
//
cl_context CreateContext()
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_context context = NULL;

    // First, select an OpenCL platform to run on.  For this example, we
    // simply choose the first available platform.  Normally, you would
    // query for all available platforms and select the most appropriate one.
    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        return NULL;
    }

    // Next, create an OpenCL context on the platform.  Attempt to
    // create a GPU-based context, and if that fails, try to create
    // a CPU-based context.
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)firstPlatformId,
        0
    };
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
                                      NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS)
    {
        std::cout << "Could not create GPU context, trying CPU..." << std::endl;
        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
                                          NULL, NULL, &errNum);
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
            return NULL;
        }
    }

    return context;
}

///
//  Create a command queue on the first device available on the
//  context
//
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device)
{
    cl_int errNum;
    cl_device_id *devices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;

    // First get the size of the devices buffer
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
        return NULL;
    }

    if (deviceBufferSize <= 0)
    {
        std::cerr << "No devices available.";
        return NULL;
    }

    // Allocate memory for the devices buffer
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
    if (errNum != CL_SUCCESS)
    {
        delete [] devices;
        std::cerr << "Failed to get device IDs";
        return NULL;
    }

    // In this example, we just choose the first available device.  In a
    // real program, you would likely use all available devices or choose
    // the highest performance device based on OpenCL device queries
    commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
    if (commandQueue == NULL)
    {
        delete [] devices;
        std::cerr << "Failed to create commandQueue for device 0";
        return NULL;
    }

    *device = devices[0];
    delete [] devices;
    return commandQueue;
}

///
//  Create an OpenCL program from the kernel source file
//
cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
    cl_int errNum;
    cl_program program;

    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open())
    {
        std::cerr << "Failed to open file for reading: " << fileName << std::endl;
        return NULL;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1,
                                        (const char**)&srcStr,
                                        NULL, NULL);
    if (program == NULL)
    {
        std::cerr << "Failed to create CL program from source." << std::endl;
        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}

///
//  Create memory objects used as the arguments to the kernel
//  The kernel takes three arguments: result (output), a (input),
//  and b (input)
//
bool CreateMemObjects(cl_context context, cl_mem memObjects[3],
                      int n, float *a, float *b)
{
    memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * n, a, NULL);
    memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * n, b, NULL);
    memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * n, NULL, NULL);

    if (memObjects[0] == NULL || memObjects[1] == NULL || memObjects[2] == NULL)
    {
        std::cerr << "Error creating memory objects." << std::endl;
        return false;
    }

    return true;
}

///
//  Cleanup any created OpenCL resources
//
void Cleanup(cl_context context, cl_program program, cl_command_queue commandQueue)
{
    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);

    if (program != 0)
        clReleaseProgram(program);

    if (context != 0)
        clReleaseContext(context);

}


int execKern(cl_program program, cl_command_queue commandQueue, cl_context context,
            const char* kName, int n,
            float* a, float *b, float *r)
{

    cl_int errNum;
    cl_mem memObjects[3] = {0,0,0};
    cl_kernel kernel = 0;

    kernel = clCreateKernel(program, kName, NULL);
    if (kernel == NULL)
    {
        std::cerr << "Error failed to create " << kName << std::endl;
        goto execKern_Err;
    }

    if (!CreateMemObjects(context, memObjects, n, a, b))
    {
        std::cerr << "Error Mem create failed in " << kName << std::endl;
        goto execKern_Err;
    }

    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error setting kernel arguments." << std::endl;
        goto execKern_Err;
    }

    size_t globalWorkSize[1];
    globalWorkSize[0] = n;
    size_t localWorkSize[1];
    localWorkSize[0] = 1;

    errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                        globalWorkSize, localWorkSize,
                                        0, NULL, NULL);
    if(errNum != CL_SUCCESS){
        std::cerr << kName << " failed to run properly" << std::endl;
        goto execKern_Err;
    }

    errNum = clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE, 0,
                                    n*sizeof(float), r, 0, NULL, NULL);
    if(errNum != CL_SUCCESS){
        std::cerr << kName << " had copy back errors" << std::endl;
        goto execKern_Err;
    }

    for(int i = 0; i < 5; i++){
        std::cout << r[i] << " ";
    }
    std::cout << std::endl;
    std::cout << kName << " run successfully" << std::endl;
    return 0;

execKern_Err:
    for(int i=0; i<3; i++){
        if(memObjects[i] != 0) 
            clReleaseMemObject(memObjects[i]);
    }
    if(kernel != 0)
        clReleaseKernel(kernel);

    return -1;
}


int main(int argc, char** argv)
{

    if(argc < 2) exit(-1);

    int n = atoi(argv[1]);
    std::cout << "Problem size: " << n << std::endl;

    cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_program program = 0;
    cl_device_id device = 0;
    cl_kernel kernel = 0;
    cl_mem memObjects[3] = { 0, 0, 0 };
    cl_int errNum;

    // Create an OpenCL context on first available platform
    context = CreateContext();
    if (context == NULL)
    {
        std::cerr << "Failed to create OpenCL context." << std::endl;
        return 1;
    }

    // Create a command-queue on the first device available
    // on the created context
    commandQueue = CreateCommandQueue(context, &device);
    if (commandQueue == NULL)
    {
        Cleanup(context, program, commandQueue);
        return 1;
    }

    program = CreateProgram(context, device, "assignment.cl");
    if (program == NULL)
    {
        Cleanup(context, program, commandQueue);
        return 1;
    }

    float *result = (float*) malloc(n*sizeof(float));
    float *a = (float*)malloc(n*sizeof(float));
    float *b = (float*)malloc(n*sizeof(float));
    for (int i = 0; i < n; i++)
    {
        a[i] = (float)i;
        b[i] = (float)(i * 2);
    }


    clock_t tic = clock();
    execKern(program, commandQueue, context, "addk",  n, a, b, result);
    execKern(program, commandQueue, context, "subk",  n, a, b, result);
    execKern(program, commandQueue, context, "mulk",  n, a, b, result);
    execKern(program, commandQueue, context, "divk",  n, a, b, result);
    execKern(program, commandQueue, context, "convk", n, a, b, result);
    clock_t toc = clock();
    printf("Elapsed time %f seconds\n", (double)(toc-tic) / CLOCKS_PER_SEC);

    Cleanup(context, program, commandQueue);

    return 0;
}

//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// Modified by lquinn

// Convolution.cpp
//
//    This is a simple example that demonstrates OpenCL platform, device, and context
//    use.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#if !defined(CL_CALLBACK)
#define CL_CALLBACK
#endif


const unsigned int i_sigWidth = 49;
const unsigned int i_sigHeight = 49;
float i_sig[i_sigHeight][i_sigWidth];

const unsigned int i_maskWidth = 7;
const unsigned int i_maskHeight = 7;
float i_mask[i_maskWidth][i_maskHeight];

// the output is inputSize - sizeof(filter) - 1 because the center of the 
// filter doesn't clip ideally if you want to clip we can do that too
const unsigned int o_sigWidth  = i_sigWidth - i_maskWidth - 1;
const unsigned int o_sigHeight  = i_sigHeight - i_maskHeight - 1;
float o_sig[o_sigHeight][o_sigWidth];

///
// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void CL_CALLBACK contextCallback(
	const char * errInfo,
	const void * private_info,
	size_t cb,
	void * user_data)
{
	std::cout << "Error occured during context use: " << errInfo << std::endl;
	// should really perform any clearup and so on at this point
	// but for simplicitly just exit.
	exit(1);
}


void genSquareMatrix(float* mat, int sizeX, int sizeY)
{
    srand(time(NULL));
    for(int y = 0; y < sizeY; y++){
        for(int x = 0; x < sizeX; x++){
            mat[y][x] = rand() % 50;
        }
    }
}
    cl_context context = NULL;
	cl_command_queue queue;
	cl_program program;


// I don't like the complex however it reduces code size signifcantly
void launchMatKernel(const char* kernelName, 
    cl_context *context, 
    cl_command_queue *queue, 
    cl_program *program,
    int sizeofSigType,  // sizeof type filter, out, and sig must all be same type
    void* sig, unsigned int sigH, unsigned int sigW,
    void* out, unsigned int outH, unsigned int outW, 
    void* filter, unsigned int filterH, unsigned int filterW
    )
{

    cl_int errNum;
	cl_kernel kernel;
	cl_mem inputSignalBuffer;
	cl_mem outputSignalBuffer;
	cl_mem maskBuffer;
	
    // Create kernel object
	kernel = clCreateKernel(
		*program,
		 kernelName,
		&errNum);
	checkErr(errNum, "clCreateKernel");
    printf("Kernel complied\n");


	// Now allocate buffers
	inputSignalBuffer = clCreateBuffer(
		*context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeofSigType * sigH * sigW,
		static_cast<void *>(sig),
		&errNum);
	checkErr(errNum, "clCreateBuffer(inputSignal)");

	maskBuffer = clCreateBuffer(
		*context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeofSigType * filterH * filterW,
		static_cast<void *>(filter),
		&errNum);
	checkErr(errNum, "clCreateBuffer(mask)");

	outputSignalBuffer = clCreateBuffer(
		*context,
		CL_MEM_WRITE_ONLY,
		sizeofSigType * outH * outW,
		NULL,
		&errNum);
	checkErr(errNum, "clCreateBuffer(outputSignal)");
    
    printf("Buffers made\n");

    errNum  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputSignalBuffer);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &maskBuffer);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputSignalBuffer);
	errNum |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &sigW);
	errNum |= clSetKernelArg(kernel, 4, sizeof(unsigned int), &filterW);
	checkErr(errNum, "clSetKernelArg");

	const size_t globalWorkSize[2] = { outW, outH };
    const size_t localWorkSize[2]  = { 1, 1 };

    printf("Argument set\n");

    // Queue the kernel up for execution across the array
    errNum = clEnqueueNDRangeKernel(
		*queue, 
		kernel, 
		2,
		NULL,
        globalWorkSize, 
		localWorkSize,
        0, 
		NULL, 
		NULL);
	checkErr(errNum, "clEnqueueNDRangeKernel");
    
    printf("Kernel ran\n");
    
	errNum = clEnqueueReadBuffer(
		*queue, 
		outputSignalBuffer, 
		CL_TRUE,
        0, 
		sizeofSigType * outH * outW, 
		out,
        0, 
		NULL, 
		NULL);
	checkErr(errNum, "clEnqueueReadBuffer");
    printf("Memcpy complete\n");

}

///
//	main() for Convoloution example
//
int main(int argc, char** argv)
{
    cl_int errNum;
    cl_uint numPlatforms;
	cl_uint numDevices;
    cl_platform_id * platformIDs;
	cl_device_id * deviceIDs;
    cl_context context = NULL;
	cl_command_queue queue;
	cl_program program;

    // First, select an OpenCL platform to run on.  
	errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
	checkErr( 
		(errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
		"clGetPlatformIDs"); 
 
	platformIDs = (cl_platform_id *)alloca(
       		sizeof(cl_platform_id) * numPlatforms);

    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr( 
	   (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
	   "clGetPlatformIDs");

	// Iterate through the list of platforms until we find one that supports
	// a CPU device, otherwise fail with an error.
	deviceIDs = NULL;
	cl_uint i;
	for (i = 0; i < numPlatforms; i++)
	{
		errNum = clGetDeviceIDs(
            platformIDs[i], 
            CL_DEVICE_TYPE_GPU, 
            0,
            NULL,
            &numDevices);
		if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
	    {
			checkErr(errNum, "clGetDeviceIDs");
        }
	    else if (numDevices > 0) 
	    {
		   	deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
			errNum = clGetDeviceIDs(
				platformIDs[i],
				CL_DEVICE_TYPE_GPU,
				numDevices, 
				&deviceIDs[0], 
				NULL);
			checkErr(errNum, "clGetDeviceIDs");
			break;
	   }
	}

	// Check to see if we found at least one CPU device, otherwise return
// 	if (deviceIDs == NULL) {
// 		std::cout << "No CPU device found" << std::endl;
// 		exit(-1);
// 	}

    // Next, create an OpenCL context on the selected platform.  
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformIDs[i],
        0
    };
    context = clCreateContext(
		contextProperties, 
		numDevices,
        deviceIDs, 
		&contextCallback,
		NULL, 
		&errNum);
	checkErr(errNum, "clCreateContext");

	std::ifstream srcFile("Convolution.cl");
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading Convolution.cl");

	std::string srcProg(
        std::istreambuf_iterator<char>(srcFile),
        (std::istreambuf_iterator<char>()));

	const char * src = srcProg.c_str();
	size_t length = srcProg.length();

	// Create program from source
	program = clCreateProgramWithSource(
		context, 
		1, 
		&src, 
		&length, 
		&errNum);
	checkErr(errNum, "clCreateProgramWithSource");

	// Build program
	errNum = clBuildProgram(
		program,
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
			program, 
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
	queue = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
	checkErr(errNum, "clCreateCommandQueue");

    genSquareMatrix((float*)i_sig, i_sigHeight, i_sigWidth);
    genSquareMatrix((float*)i_mask, i_maskHeight, i_maskWidth); 

    launchMatKernel("convolveManhattan", &context, &queue, &program,
        sizeof(float),
        i_sig, i_sigHeight, i_sigWidth,
        o_sig, o_sigHeight, o_sigWidth,
        i_mask, i_maskHeight, i_maskWidth);


    printf("Kernel done?\n");
    for(int y = 0; y < o_sigHeight; y++){
        for(int x = 0; x < o_sigWidth; x++){
            printf("%f ", o_sig[y][x]);
        }
        printf("\n");
    }


    std::cout << std::endl << "Executed program succesfully." << std::endl;

	return 0;
}

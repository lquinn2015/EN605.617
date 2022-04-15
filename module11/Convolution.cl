//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// Convolution.cl
//
//    This is a simple kernel performing convolution.

__kernel void convolve(
	const __global  uint * const input,
    __constant uint * const mask,
    __global  uint * const output,
    const int inputWidth,
    const int maskWidth)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    float sum = 0;
    for (int r = 0; r < maskWidth; r++)
    {
        const int idxIntmp = (y + r) * inputWidth + x;

        for (int c = 0; c < maskWidth; c++)
        {
            float mC = 1 / (abs(maskWidth/2 - r) + abs(maskWidth/2 - c));
			sum += mask[(r * maskWidth)  + c] * input[idxIntmp + c] * mC;
        }
    } 
    
	output[y * get_global_size(0) + x] = sum;
}

__kernel void convolveManhattan(
	const __global  float * const input,
    __constant float * const mask,
    __global  float * const output,
    const int inputWidth,
    const int maskWidth)
{
    const int o_x = get_global_id(0);
    const int o_y = get_global_id(1);


    // Manhattan distance between to vectorss is abs(x_1-x_2) + abs(y_1-y_2)
    // Conovlution o_yx = sum_{r=0}^{mH} sum_{c=0}^{mW} m[r][c] * 


    uint sum = 0;
    for (int r = 0; r < maskWidth; r++)
    {
        const int idxIntmp = (y + r) * inputWidth + x;

        for (int c = 0; c < maskWidth; c++)
        {
            float scale = 1/(abs() + abs())
			sum += mask[(r * maskWidth)  + c] * input[idxIntmp + c];
        }
    } 
    
	output[y * get_global_size(0) + x] = sum ;
}

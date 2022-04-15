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

    uint sum = 0;
    for (int r = 0; r < maskWidth; r++)
    {
        const int idxIntmp = (y + r) * inputWidth + x;

        for (int c = 0; c < maskWidth; c++)
        {
			sum += mask[(r * maskWidth)  + c] * input[idxIntmp + c];
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
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    float sum = 0;
    for (int r = 0; r < maskWidth; r++)
    {
        const int idxIntmp = (y + r) * inputWidth + x;

        for (int c = 0; c < maskWidth; c++)
        {
            float scale = 1.0;
            if(r != maskWidth/2 || c != maskWidth/2)
                scale = 1 / (abs(r - maskWidth/2) + abs(c - maskWidth/2));
            
			sum += mask[(r * maskWidth)  + c] * input[idxIntmp + c] * scale;
        }
    } 
    
	output[y * get_global_size(0) + x] = sum;
}

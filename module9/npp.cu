#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_cuda.h>
#include <helper_string.h>
#include <cuda_utils.cuh>


int main(int argc, char* argv[]){


    try{
        std::string sFilename;
        

        sFilename = "lena.pgm"; // this is hardcoded because i cannot
                                // easily share large bins via git
        
        int file_errors = 0;
        std::ifstream infile(sFilename.data(), std::ifstream::in);
        
        if(infile.good()){
            std::cout << "We loaded: " << sFilename.data() << " successfully" << std::endl;
            file_errors = 0;
        }else {
            std::cout << "Loading failed " << sFilename.data() << " try again" << std::endl; 
            file_errors++;
        }
        infile.close();

        if(file_errors > 0) {
            exit(EXIT_FAILURE);
        }

        std::string sResultFilename = sFilename;

        std::string::size_type dot = sResultFilename.rfind('.');

        if (dot != std::string::npos) {
            sResultFilename = sResultFilename.substr(0, dot);
        }

        sResultFilename += "_cannyEdgeDetection.pgm";

        // declear 8 bit grayscale image
        npp::ImageCPU_8u_C1 oHostSrc;

        // read and load image to GPU
        npp::loadImage(sFilename, oHostSrc);
        npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);
        
        NppiSize oSrcSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
        NppiPoint oSrcOffset = {0,0};
        
        // we are interested in everything
        NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};

        npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);

        int nBufferSize = 0;
        Npp8u *pScratchBufferNPP = 0;

        // alloc the scratch buffer
        NPP_CHECK_NPP(nppiFilterCannyBorderGetBufferSize(oSizeROI, &nBufferSize));
        checkCuda( cudaMalloc((void**)&pScratchBufferNPP, nBufferSize) );
        
        Npp16s nLowThreshold = 12451;
        Npp16s nHighThreshold = 2621; 
    
        // run the filter
        if((nBufferSize > 0) && (pScratchBufferNPP != 0)) {
            NPP_CHECK_NPP(nppiFilterCannyBorder_8u_C1R(
                oDeviceSrc.data(), oDeviceSrc.pitch(), oSrcSize, oSrcOffset,
                oDeviceDst.data(), oDeviceDst.pitch(), oSizeROI, NPP_FILTER_SOBEL,
                NPP_MASK_SIZE_3_X_3, nLowThreshold, nHighThreshold, nppiNormL2,
                NPP_BORDER_REPLICATE, pScratchBufferNPP));
        }
        
        checkCuda( cudaFree(pScratchBufferNPP) );

        // save back to host
        npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
        oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

        saveImage(sResultFilename, oHostDst);
        std::cout << "Saved image: " << sResultFilename << std::endl; 

        nppiFree(oDeviceSrc.data());
        nppiFree(oDeviceDst.data());

        exit(EXIT_SUCCESS);
    } catch (npp::Exception &rException) {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rException << std::endl;
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
    } catch (...) {
        std::cerr << "Program error! An unknow type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
        return -1;
    }

    return 0;

}


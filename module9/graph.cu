#include <stdio.h>
#include <stdlib.h>
#include "cuda_utils.cuh"

#include <cuda_runtime.h>
#include "nvgraph.h"

void check(nvgraphStatus_t status){
    if(status != NVGRAPH_STATUS_SUCCESS){
        printf("Error : %d\n", status);
        exit(0);
    }
}


void readGraph(FILE *fp, float *val, int *dest, int *src)
{

    char *line = NULL;
    size_t amt = 0;
    float w_i; int w_idx=0;
    int d_l=-1, d_i, d_idx = 0;
    int s_i, s_idx=0;
    
    while(getline(&line, &amt, fp) != -1){ // while we can get another line
        
        sscanf(line, "%f %d %d", &w_i, &s_i, &d_i); // parse line
        if(d_idx == 0 || d_l != d_i)
        {
            dest[d_idx++] = d_i;
            d_l = d_i;  
        }
        val[w_idx++] = w_i;
        src[s_idx++] = s_i;
        printf("%d->%d  with weight %f\n", s_i, d_i, w_i);
    }
    printf("w_idx=s_idx=%d, and d_idx=%d\n", w_idx, d_idx);
    return;
}


void sssp_graph(const char* fname)
{
    printf("Starting sssp\n");
    FILE* fp = fopen(fname, "r");
    size_t linesize = 0;
    char* line = NULL;

    int len = getline(&line, &linesize, fp); // reads one line
    if(len == -1){
        exit(-1); //error
    }
    printf("%s\n", line); 

    int n, nnz, ccol, vertex_numsets = 1, edge_numsets = 1;
    float *sssp_1_h;
    void **vertex_dim;
    sscanf((const char*)line, "%d %d %d", &n, &nnz, &ccol);
    printf("Graph #vert=%d, #edges=%d\n", n, nnz);
    nnz *=2; 
    //nvgraph varibles
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCSCTopology32I_t CSC_input;
    cudaDataType_t edge_dimT = CUDA_R_32F;
    cudaDataType_t* vertex_dimT;

    // init data
    sssp_1_h = (float*)malloc(n*sizeof(float));
    vertex_dim = (void**)malloc(vertex_numsets*sizeof(void*));
    vertex_dimT = (cudaDataType_t*)malloc(vertex_numsets*sizeof(cudaDataType_t));
    CSC_input = (nvgraphCSCTopology32I_t) malloc(sizeof(struct nvgraphCSCTopology32I_st));
    vertex_dim[0] = (void*) sssp_1_h; vertex_dimT[0] = CUDA_R_32F;
    
    float *weights = (float*) malloc(nnz * sizeof(float));
    int *dest = (int*) malloc((ccol+1)*sizeof(float));
    int *src = (int*) malloc(nnz*sizeof(float));
    readGraph(fp, weights, dest, src);
   
    dest[ccol] = nnz;
 
    printf("Graph IO complete running nvgraph now\n");
     
    check( nvgraphCreate(&handle));
    check( nvgraphCreateGraphDescr(handle, &graph));
    CSC_input->nvertices = n; CSC_input->nedges = nnz;
    CSC_input->destination_offsets = dest;
    CSC_input->source_indices = src;

    // Set connectivity and properties
    check( nvgraphSetGraphStructure(handle, graph, (void*)CSC_input, NVGRAPH_CSC_32));
    check( nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dimT));
    check( nvgraphAllocateEdgeData(handle, graph, edge_numsets, &edge_dimT));
    check( nvgraphSetEdgeData(handle, graph, (void*)weights, 0));
    
    // solve
    int src_vert = 0;
    check( nvgraphSssp(handle, graph, 0, &src_vert, 0));
    // get and print results
    check(nvgraphGetVertexData(handle, graph, (void*)sssp_1_h, 0));
    for(int x = 0; x < n; x++){
        printf("Cost to get from 0->%d was %f\n", x, sssp_1_h[x]);
    }

    // free data
    free(sssp_1_h); free(vertex_dim);
    free(vertex_dimT); free(CSC_input);
    check(nvgraphDestroyGraphDescr(handle, graph));
    check(nvgraphDestroy(handle));
    free(weights);    free(dest);
    free(src);     free(line);
    fclose(fp);
    return;
}

int main()
{    
    sssp_graph("csc1.lsv");
    //sssp_graph("csc2.lsv");
    //sssp_graph("csc3.lsv");
    return 0;
}

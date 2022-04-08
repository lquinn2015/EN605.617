

__kernel void addk( __global const float *a,
                    __global const float *b,
                    __global float *res)
{

    int gid = get_global_id(0);
    res[gid] = a[gid] + b[gid];
}

__kernel void subk( __global const float *a,
                    __global const float *b,
                    __global float *res)
{

    int gid = get_global_id(0);
    res[gid] = a[gid] - b[gid];
}

__kernel void mulk( __global const float *a,
                    __global const float *b,
                    __global float *res)
{

    int gid = get_global_id(0);
    res[gid] = a[gid] * b[gid];
}

__kernel void divk( __global const float *a,
                    __global const float *b,
                    __global float *res)
{

    int gid = get_global_id(0);
    res[gid] = a[gid] / b[gid];
}

__kernel void convk( __global const float *a,
                    __global const float *b,
                    __global float *res)
{

    int gid = get_global_id(0);
    res[gid] = a[gid]*b[gid] + a[gid];
}

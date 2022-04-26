//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Dan Ginsburg, Timothy Mattson
// ISBN-10:   ??????????
// ISBN-13:   ?????????????
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/??????????
//            http://www.????????.com
//

// simple.cl
//
//    This is a simple example demonstrating buffers and sub-buffer usage

__kernel void square(__global * buffer)
{
	size_t id = get_global_id(0);
	buffer[id] = buffer[id] * buffer[id];
}

__kernel void average(__global * buffer, int n)
{
	size_t id = get_global_id(0);
    int i = id;
    int acc = 0;
    int c = 0;
    for(i = id; i < id+4 && i < n; i++){
        
        acc += buffer[i];
        c++;
    }
    buffer[id] = acc/c;
}

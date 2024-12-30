// #include "cnn.h"

// char *ReadKenelSourceFile(const char *filename, size_t *length) {
//     FILE *file = NULL;
//     size_t sourceLength;
//     char * sourceString;
//     int ret;
//     file = fopen(filename, "rb");
//     if(file == NULL) {
//         printf("..xxx:");
//         return NULL;
//     }
//     fseek(file, 0, SEEK_END);
//     sourceLength = ftell(file);
//     fseek(file, 0, SEEK_SET);
//     sourceString = (char*) malloc(sourceLength + 1);
//     sourceString[0] = '\0';
//     ret = fread(sourceString, sourceLength, 1, file);
//     if(ret == 0) {
//         printf("...xx, %d\n", ferror(file));
//         return NULL;
//     }
//     fclose(file);
//     if(length != 0) {
//         *length = sourceLength;
//     }
//     sourceString[sourceLength] = '\0';
//     return sourceString;

// }

// bool CNN::opencl_init() {
//     int chan = 6;
//     int conSize = 25 * 6 *sizeof(float);
//     int N = 32;
//     int M = 32;
//     int N1 = 28;
//     int M1 = 28;
//     int len_bi = 6 *sizeof(float);
//     // insert code here...
//     int datasize = N * M *sizeof(float);

//     int datasoze = chan * N1 * M1 *sizeof(float);
//     size_t programLength;
//     char * source = ReadKenelSourceFile("./kernel/kernel.cl", &programLength);
    

//     //-----------------------------------------------------
//     // STEP 1: Discover and initialize the platforms
//     //-----------------------------------------------------
//     // Use clGetPlatformIDs() to retrieve the number of
//     // platforms
//     status = clGetPlatformIDs(0, NULL, &numPlatforms);

//     // Allocate enough space for each platform
//     platforms = (cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id));

//     // Fill in platforms with clGetPlatformIDs()
//     status = clGetPlatformIDs(numPlatforms, platforms, NULL);

//     //-----------------------------------------------------
//     // STEP 2: Discover and initialize the devices
//     //-----------------------------------------------------
//     // Use clGetDeviceIDs() to retrieve the number of
//     // devices present
//     status = clGetDeviceIDs(platforms[0],CL_DEVICE_TYPE_GPU,0,NULL,&numDevices);

//     // Allocate enough space for each device
//     devices = (cl_device_id*)malloc(numDevices*sizeof(cl_device_id));

//     // Fill in devices with clGetDeviceIDs()
//     status = clGetDeviceIDs(platforms[0],CL_DEVICE_TYPE_GPU,numDevices,devices,NULL);

//     size_t size;
//     status = clGetDeviceInfo(devices[0], CL_DEVICE_NAME, 0, NULL, &size);
//     char *DName = (char*)malloc(size);
//     status = clGetDeviceInfo(devices[0], CL_DEVICE_NAME, size, DName, NULL);
//     // printf("DEVICE NAME: %s \n", DName);

//     //-----------------------------------------------------
//     // STEP 3: Create a context
//     //-----------------------------------------------------

//     // Create a context using clCreateContext() and
//     // associate it with the devices
//     context = clCreateContext(NULL,numDevices,devices,NULL,NULL,&status);
//     if(status != CL_SUCCESS) std::cout <<"context" << status << std::endl;
//     //-----------------------------------------------------
//     // STEP 4: Create a command queue
//     //-----------------------------------------------------

//     // Create a command queue using clCreateCommandQueue(),
//     // and associate it with the device you want to execute
//     // on
//     cmdQueue = clCreateCommandQueue(context,devices[0],CL_QUEUE_PROFILING_ENABLE,&status);
//     if(status != CL_SUCCESS) std::cout <<"cmd" << status << std::endl;

//     //-----------------------------------------------------
//     // STEP 5: Create device buffers
//     //-----------------------------------------------------
//     cl_neuron_input = clCreateBuffer(context,CL_MEM_READ_WRITE,datasize,NULL,&status);
//     cl_weight_C1 = clCreateBuffer(context,CL_MEM_READ_WRITE,conSize,NULL,&status);
//     cl_neuron_C1 = clCreateBuffer(context,CL_MEM_READ_WRITE,datasoze,NULL,&status);
//     cl_bias_C1 = clCreateBuffer(context,CL_MEM_READ_WRITE,len_bi,NULL,&status);
//     cl_neuron_S2 = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*num_neuron_S2_CNN,NULL,&status);
// 	cl_neuron_C3 = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*num_neuron_C3_CNN,NULL,&status);
// 	cl_neuron_S4 = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*num_neuron_S4_CNN,NULL,&status);
// 	cl_neuron_C5 = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*num_neuron_C5_CNN,NULL,&status);
// 	cl_neuron_output = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*num_neuron_output_CNN,NULL,&status);
// 	cl_weight_S2 = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*len_weight_S2_CNN,NULL,&status);
// 	cl_mem cl_bias_S2 = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*len_bias_S2_CNN,NULL,&status);
// 	cl_weight_C3 = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*len_weight_C3_CNN,NULL,&status);
// 	cl_mem cl_bias_C3 = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*len_bias_C3_CNN,NULL,&status);
// 	cl_weight_S4 = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*len_weight_S4_CNN,NULL,&status);
// 	cl_mem cl_bias_S4 = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*len_bias_S4_CNN,NULL,&status);
// 	cl_weight_C5 = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*len_weight_C5_CNN,NULL,&status);
// 	cl_mem cl_bias_C5 = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*len_bias_C5_CNN,NULL,&status);
// 	cl_mem cl_weight_output = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*len_weight_output_CNN,NULL,&status);
// 	cl_mem cl_bias_output = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*len_bias_output_CNN,NULL,&status);
//     //-----------------------------------------------------
//     // STEP 6: Write host data to device buffers
//     //-----------------------------------------------------

//     // Use clEnqueueWriteBuffer() to write input array A to
//     // the device buffer bufferA
//     status = clEnqueueWriteBuffer(cmdQueue,cl_neuron_input,CL_FALSE,0,datasize,neuron_input,0,NULL,NULL);
//     if(status != CL_SUCCESS) std::cout <<"A" << status << std::endl;
//     // Use clEnqueueWriteBuffer() to write input array B to
//     // the device buffer bufferB
//     status = clEnqueueWriteBuffer(cmdQueue,cl_weight_C1,CL_FALSE,0,conSize,weight_C1,0,NULL,NULL);
//     if(status != CL_SUCCESS) std::cout <<"B" << status << std::endl;
//     status = clEnqueueWriteBuffer(cmdQueue,cl_bias_C1,CL_FALSE,0,len_bi,bias_C1,0,NULL,NULL);
//     if(status != CL_SUCCESS) std::cout <<"bi" << status << std::endl;
//     // cout << "p-:" << status << endl;
//     //-----------------------------------------------------
//     // STEP 7: Create and compile the program
//     //-----------------------------------------------------

//     // Create a program using clCreateProgramWithSource()
//     program_forward = clCreateProgramWithSource(context,1,(const char**)&source,NULL,&status);
//     // cout << "p:" << status << endl;
//     // Build (compile) the program for the devices with
//     // clBuildProgram()
//     status = clBuildProgram(program_forward,numDevices,devices,NULL,NULL,NULL);
//     // cout << status << endl;

//     //---------------------------------------------------
//     // 编译信息
//     //----------------------------------------------------
//     // size_t logsize;
//     // status = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &logsize);
//     // char *buffer = (char *)malloc(logsize* sizeof(char));
//     // status = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, logsize, buffer, NULL);
//     // printf("%s\n", buffer);

//     //-----------------------------------------------------
//     // STEP 8: Create the kernel
//     //-----------------------------------------------------

    

//     // Use clCreateKernel() to create a kernel from the
//     // vector addition function (named "vecadd")
//     kernel_forward_c1 = clCreateKernel(program_forward,"kernel_forward_c1",&status);
//     // cout << status << endl;
//     status = clSetKernelArg(kernel_forward_c1, 0, sizeof(cl_mem), (void*) &cl_neuron_input);
//     status = clSetKernelArg(kernel_forward_c1, 1, sizeof(cl_mem), (void*) &cl_weight_C1);
//     status = clSetKernelArg(kernel_forward_c1, 2, sizeof(cl_mem), (void*) &cl_bias_C1);
//     status = clSetKernelArg(kernel_forward_c1, 3, sizeof(cl_mem), (void*) &cl_neuron_C1);
//     int channel = 0;
//     status = clSetKernelArg(kernel_forward_c1, 4, sizeof(int), (void*) &channel);
//     int out_width = 28;
//     status = clSetKernelArg(kernel_forward_c1, 5, sizeof(int), (void*) &out_width);
//     int out_height=28;
//     status = clSetKernelArg(kernel_forward_c1, 6, sizeof(int), (void*) &out_height);
//     int kernel_width=5;
//     status = clSetKernelArg(kernel_forward_c1, 7, sizeof(int), (void*) &kernel_width);
//     int kernel_height=5;
//     status = clSetKernelArg(kernel_forward_c1, 8, sizeof(int), (void*) &kernel_height);
//     int in_num=1;
//     status = clSetKernelArg(kernel_forward_c1, 9, sizeof(int), (void*) &in_num);
//     int in_width=32;
//     status = clSetKernelArg(kernel_forward_c1, 10, sizeof(int), (void*) &in_width);
//     int in_height=32;
//     status = clSetKernelArg(kernel_forward_c1, 11, sizeof(int), (void*) &in_height);




    
//     const int wg_dim = 8;
//     const size_t global[3] = {(size_t)(chan+2), (size_t)N,(size_t)M}; // normal, local
// //    const size_t global[2] = {(size_t)N / 4,(size_t)M / 4}; // v
// //    const size_t global[2] = {(size_t)N / 8,(size_t)M / 8}; // mul, pipeline
//     const size_t local[3] = {2, 8, 8};
//     // cl_event event;
//     status = clEnqueueNDRangeKernel(cmdQueue, kernel_forward_c1, 3, NULL, global, local, 0, NULL, NULL);
//     if(status != CL_SUCCESS) std::cout <<"k" << status << std::endl;
//     //    status = clWaitForEvents(1, &event);

//     // clFinish(cmdQueue);
//     // cl_ulong time_start;
//     // cl_ulong time_end;

//     // clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
//     // clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

//     // double nanoSeconds = time_end-time_start;
//     // printf("OpenCl Execution time is: %0.3f ms \n",nanoSeconds/1000000);

//     //-----------------------------------------------------
//     // STEP 12: Read the output buffer back to the host
//     //-----------------------------------------------------

//     // Use clEnqueueReadBuffer() to read the OpenCL output
//     // buffer (cl_neuron_C1)
//     // to the host output array (C)
//     clEnqueueReadBuffer(
//         cmdQueue,
//         cl_neuron_C1,
//         CL_TRUE,
//         0,
//         datasoze,
//         neuron_C1,
//         0,
//         NULL,
//         NULL);
//     if(status != CL_SUCCESS) std::cout <<"C" << status << std::endl;
//     // Verify the output
// //    bool result = true;
// //    for(int i = 0; i < ele; i++) {
// //        if(C[i] != C1[i]) {
// //            printf("c:%f, c1:%f, i:%d \n", C[i], C1[i], i);
// //            result = false;
// //            break;
// //        }
// //    }
// //    if(result) {
// //        printf("Output is correct\n");
// //    } else {
// //        printf("Output is incorrect\n");
// //    }

//     //-----------------------------------------------------
//     // STEP 13: Release OpenCL resources
//     //-----------------------------------------------------

//     // Free OpenCL resources
//    clReleaseKernel(kernel_forward_c1);
//    clReleaseProgram(program_forward);
//    clReleaseCommandQueue(cmdQueue);
//    clReleaseMemObject(cl_neuron_input);
//    clReleaseMemObject(cl_weight_C1);
//    clReleaseMemObject(cl_neuron_C1);
//    clReleaseMemObject(cl_bias_C1);
//    clReleaseContext(context);
//     // cout << C1[0] << endl;
//     // cout << C[0] << endl;
//     // checkResult(1022 * 1022, C, C1);
//     // return 0;
// 	// for (int channel = 0; channel < num_map_C1_CNN; channel++) {
// 	// 	for (int y = 0; y < height_image_C1_CNN; y++) {
// 	// 		for (int x = 0; x < width_image_C1_CNN; x++) {
// 	// 			int index = (channel*height_image_C1_CNN*width_image_C1_CNN) + y*width_image_C1_CNN + x;  //当前神经元
// 	// 			neuron_C1[index] = 0.0;
// 	// 			//卷积运算
// 	// 			for (int inc = 0; inc < num_map_input_CNN; inc++) {
// 	// 				int addr1 = get_index(0, 0, num_map_input_CNN * channel + inc, width_kernel_conv_CNN, height_kernel_conv_CNN, num_map_C1_CNN * num_map_input_CNN);
// 	// 				int addr2 = get_index(0, 0, inc, width_image_input_CNN, height_image_input_CNN, num_map_input_CNN);
// 	// 				const float* pw = &weight_C1[0] + addr1;       //卷积核
// 	// 				const float* pi = data_single_image + addr2;   //输入图像
// 	// 				float sum = 0.0;
// 	// 				const float* ppw = pw;
// 	// 				const float* ppi = pi + y * width_image_input_CNN + x;
// 	// 				for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
// 	// 					for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
// 	// 						sum += *ppw++ * ppi[wy * width_image_input_CNN + wx];
// 	// 					}
// 	// 				}
// 	// 				neuron_C1[index] += sum;
// 	// 			}
// 	// 			neuron_C1[index] += bias_C1[channel];     //加偏置
// 	// 			neuron_C1[index] = activation_function_tanh(neuron_C1[index]);  //激励函数
// 	// 		}
// 	// 	}
// 	// }
//     // memcpy(neuron_C1, C, datasoze);
// 	return true;
// }
// // using namespace std;

// // connection table [Y.Lecun, 1998 Table.1]
// #define O true
// #define X false
// static const bool tbl[6][16] = {
// 	O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
// 	O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
// 	O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
// 	X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
// 	X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
// 	X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
// };
// #undef O
// #undef X

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


// bool CNN::Forward_C1()
// {
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
//     // float test[1024 * 1024];
//     // float filter[] = {0,1,0,1,0,1,0,1,0};
//     // float out1[0];
// //    float out2[1022 * 1022];
//     // float *A = NULL;  // Input array
//     // float *B = NULL;  // Input array
//     // float *C = NULL;  // Output array
//     // float *bi = NULL;
//     // // float *C1 = NULL;
//     // // Allocate space for input/output data
//     // A = (float*)malloc(datasize);
//     // B = (float*)malloc(conSize);
//     // C = (float*)malloc(datasoze);
//     // bi = (float*)malloc(len_bi);
//     // cout<< 0 <<endl;
//     // memcpy(A, neuron_input, datasize);
//     // // cout<< 1 <<endl;
//     // memcpy(B, weight_C1, conSize);
//     // // cout<< 2 <<endl;
//     // memcpy(bi, bias_C1, len_bi);
// //    convolutionSerialBlocking<3, 2, 2, float>(1024, 1024, test, filter, out);
// //    convolutionSerialBlockingAVX<3, 2, 2>(1024, 1024, A, B, C1);
// //    checkResult(1024 * 1024, C, C1);
// //    for(int i = 0; i < 8; i++) {
// //        for(int j = 0; j < 8; j++) {
// //            std::cout << test[i * 8 + j] << " ";
// //        }
// //        std::cout << std::endl;
// //    }
// //    for(int i = 0; i < 6; i++) {
// //        for(int j = 0; j < 6; j++) {
// //            std::cout << out[i * 6 + j] << " ";
// //        }
// //        std::cout << std::endl;
// //    }
// //    std::cout << "Hello, World!\n";
//     size_t programLength;
//     char * source = ReadKenelSourceFile("./kernel/kernel.cl", &programLength);
//     // cout << source[0] << endl;
//     cl_int status;

//     //-----------------------------------------------------
//     // STEP 1: Discover and initialize the platforms
//     //-----------------------------------------------------

//     cl_uint numPlatforms = 0;
//     cl_platform_id *platforms = NULL;

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

//     cl_uint numDevices = 0;
//     cl_device_id *devices = NULL;

//     // Use clGetDeviceIDs() to retrieve the number of
//     // devices present
//     status = clGetDeviceIDs(
//         platforms[0],
//         CL_DEVICE_TYPE_GPU,
//         0,
//         NULL,
//         &numDevices);

//     // Allocate enough space for each device
//     devices = (cl_device_id*)malloc(numDevices*sizeof(cl_device_id));

//     // Fill in devices with clGetDeviceIDs()
//     status = clGetDeviceIDs(
//         platforms[0],
//         CL_DEVICE_TYPE_GPU,
//         numDevices,
//         devices,
//         NULL);

//     size_t size;
//     status = clGetDeviceInfo(devices[0], CL_DEVICE_NAME, 0, NULL, &size);
//     char *DName = (char*)malloc(size);
//     status = clGetDeviceInfo(devices[0], CL_DEVICE_NAME, size, DName, NULL);
//     // printf("DEVICE NAME: %s \n", DName);

//     //-----------------------------------------------------
//     // STEP 3: Create a context
//     //-----------------------------------------------------

//     cl_context context = NULL;

//     // Create a context using clCreateContext() and
//     // associate it with the devices
//     context = clCreateContext(
//         NULL,
//         numDevices,
//         devices,
//         NULL,
//         NULL,
//         &status);
//     if(status != CL_SUCCESS) std::cout <<"context" << status << std::endl;
//     //-----------------------------------------------------
//     // STEP 4: Create a command queue
//     //-----------------------------------------------------

//     cl_command_queue cmdQueue;

//     // Create a command queue using clCreateCommandQueue(),
//     // and associate it with the device you want to execute
//     // on
//     cmdQueue = clCreateCommandQueue(
//         context,
//         devices[0],
//         CL_QUEUE_PROFILING_ENABLE,
//         &status);
//     if(status != CL_SUCCESS) std::cout <<"cmd" << status << std::endl;

//     //-----------------------------------------------------
//     // STEP 5: Create device buffers
//     //-----------------------------------------------------

//     cl_mem bufferA;  // Input array on the device
//     cl_mem bufferB;  // Input array on the device
//     cl_mem bufferC;  // Output array on the device
//     cl_mem bufferbi;

//     // Use clCreateBuffer() to create a buffer object (d_A)
//     // that will contain the data from the host array A
//     bufferA = clCreateBuffer(
//         context,
//         CL_MEM_READ_ONLY,
//         datasize,
//         NULL,
//         &status);

//     // Use clCreateBuffer() to create a buffer object (d_B)
//     // that will contain the data from the host array B
//     bufferB = clCreateBuffer(
//         context,
//         CL_MEM_READ_ONLY,
//         conSize,
//         NULL,
//         &status);

//     // Use clCreateBuffer() to create a buffer object (d_C)
//     // with enough space to hold the output data
//     bufferC = clCreateBuffer(
//         context,
//         CL_MEM_WRITE_ONLY,
//         datasoze,
//         NULL,
//         &status);

//     bufferbi = clCreateBuffer(
//         context,
//         CL_MEM_READ_ONLY,
//         len_bi,
//         NULL,
//         &status);

//     //-----------------------------------------------------
//     // STEP 6: Write host data to device buffers
//     //-----------------------------------------------------

//     // Use clEnqueueWriteBuffer() to write input array A to
//     // the device buffer bufferA
//     status = clEnqueueWriteBuffer(
//         cmdQueue,
//         bufferA,
//         CL_FALSE,
//         0,
//         datasize,
//         neuron_input,
//         0,
//         NULL,
//         NULL);
//     if(status != CL_SUCCESS) std::cout <<"A" << status << std::endl;
//     // Use clEnqueueWriteBuffer() to write input array B to
//     // the device buffer bufferB
//     status = clEnqueueWriteBuffer(
//         cmdQueue,
//         bufferB,
//         CL_FALSE,
//         0,
//         conSize,
//         weight_C1,
//         0,
//         NULL,
//         NULL);
//     if(status != CL_SUCCESS) std::cout <<"B" << status << std::endl;
//     status = clEnqueueWriteBuffer(
//         cmdQueue,
//         bufferbi,
//         CL_FALSE,
//         0,
//         len_bi,
//         bias_C1,
//         0,
//         NULL,
//         NULL);
//     if(status != CL_SUCCESS) std::cout <<"bi" << status << std::endl;
//     // cout << "p-:" << status << endl;
//     //-----------------------------------------------------
//     // STEP 7: Create and compile the program
//     //-----------------------------------------------------

//     // Create a program using clCreateProgramWithSource()
//     cl_program program = clCreateProgramWithSource(
//         context,
//         1,
//         (const char**)&source,
//         NULL,
//         &status);
//     // cout << "p:" << status << endl;
//     // Build (compile) the program for the devices with
//     // clBuildProgram()
//     status = clBuildProgram(
//         program,
//         numDevices,
//         devices,
//         NULL,
//         NULL,
//         NULL);
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

//     cl_kernel kernel = NULL;

//     // Use clCreateKernel() to create a kernel from the
//     // vector addition function (named "vecadd")
//     kernel = clCreateKernel(program,"kernel_forward_c1",&status);
//     // cout << status << endl;
//     status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &bufferA);
//     status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &bufferB);
//     status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*) &bufferbi);
//     status = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*) &bufferC);
//     int channel = 0;
//     status = clSetKernelArg(kernel, 4, sizeof(int), (void*) &channel);
//     int out_width = 28;
//     status = clSetKernelArg(kernel, 5, sizeof(int), (void*) &out_width);
//     int out_height=28;
//     status = clSetKernelArg(kernel, 6, sizeof(int), (void*) &out_height);
//     int kernel_width=5;
//     status = clSetKernelArg(kernel, 7, sizeof(int), (void*) &kernel_width);
//     int kernel_height=5;
//     status = clSetKernelArg(kernel, 8, sizeof(int), (void*) &kernel_height);
//     int in_num=1;
//     status = clSetKernelArg(kernel, 9, sizeof(int), (void*) &in_num);
//     int in_width=32;
//     status = clSetKernelArg(kernel, 10, sizeof(int), (void*) &in_width);
//     int in_height=32;
//     status = clSetKernelArg(kernel, 11, sizeof(int), (void*) &in_height);
// //    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &bufferC);
// //    size_t workgroupSize = 0;
// //        status = clGetKernelWorkGroupInfo(kernel, devices[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(workgroupSize), &workgroupSize, NULL);
// //        if(status != CL_SUCCESS)
// //        {
// //            puts("Query max workgroup size failed!");
// //            return 0;
// //        }
// //    printf("Current work-group size: %zu\n", workgroupSize);
//     const int wg_dim = 8;
//     const size_t global[3] = {(size_t)(chan+2), (size_t)N,(size_t)M}; // normal, local
// //    const size_t global[2] = {(size_t)N / 4,(size_t)M / 4}; // v
// //    const size_t global[2] = {(size_t)N / 8,(size_t)M / 8}; // mul, pipeline
//     const size_t local[3] = {2, 8, 8};
//     // cl_event event;
//     status = clEnqueueNDRangeKernel(cmdQueue, kernel, 3, NULL, global, local, 0, NULL, NULL);
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
//     // buffer (bufferC)
//     // to the host output array (C)
//     clEnqueueReadBuffer(
//         cmdQueue,
//         bufferC,
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
//    clReleaseKernel(kernel);
//    clReleaseProgram(program);
//    clReleaseCommandQueue(cmdQueue);
//    clReleaseMemObject(bufferA);
//    clReleaseMemObject(bufferB);
//    clReleaseMemObject(bufferC);
//    clReleaseMemObject(bufferbi);
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


// bool CNN::Forward_S2()
// {
// 	float scale_factor = 1.0 / (width_kernel_pooling_CNN * height_kernel_pooling_CNN);

// 	for (int i=0; i<num_map_S2_CNN; i++) {
// 		int block = width_image_C1_CNN * height_image_C1_CNN * i;
// 		for (int y=0; y<height_image_S2_CNN; y++) {
// 			for (int x=0; x<width_image_S2_CNN; x++) {
// 				int rows = y * width_kernel_pooling_CNN;
// 				int cols = x * height_kernel_pooling_CNN;
// 				int index = (i*height_image_S2_CNN*width_image_S2_CNN) + y*width_image_S2_CNN + x;

//                 neuron_S2[index] = 0.0;
// 				for (int m = 0; m < width_kernel_pooling_CNN; m++) {
// 					for (int n = 0; n < height_kernel_pooling_CNN; n++) {
//                         neuron_S2[index] += weight_S2[i] * neuron_C1[(rows + m) * width_image_C1_CNN + cols + n + block];
// 					}
// 				}
// 				//
// 				neuron_S2[index] *= scale_factor;
// 				neuron_S2[index] += bias_S2[i] ;
// 				neuron_S2[index] = activation_function_tanh(neuron_S2[index]);
// 			}
// 		}
// 	}
// 	return true;
// }

// bool CNN::Forward_C3()
// {
// 	for (int channel = 0; channel < num_map_C3_CNN; channel++) {
// 		for (int y = 0; y < height_image_C3_CNN; y++) {
// 			for (int x = 0; x < width_image_C3_CNN; x++) {
// 				int index = (channel*height_image_C3_CNN*width_image_C3_CNN) + y*width_image_C3_CNN + x;  //当前神经元
// 				neuron_C3[index] = 0.0;
// 				//卷积运算
// 				for (int inc = 0; inc < num_map_S2_CNN; inc++) {
// 					if (!tbl[inc][channel]) continue;
// 					int addr1 = get_index(0, 0, num_map_S2_CNN * channel + inc, width_kernel_conv_CNN, height_kernel_conv_CNN, num_map_C3_CNN * num_map_S2_CNN);
// 					int addr2 = get_index(0, 0, inc, width_image_S2_CNN, height_image_S2_CNN, num_map_S2_CNN);  //输入图像
// 					const float* pw = &weight_C3[0] + addr1;   //卷积核
// 					const float* pi = &neuron_S2[0] + addr2;   //输入图像
// 					float sum = 0.0;
// 					const float* ppw = pw;
// 					const float* ppi = pi + y * width_image_S2_CNN + x;
// 					for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
// 						for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
// 							sum += *ppw++ * ppi[wy * width_image_S2_CNN + wx];
// 						}
// 					}
// 					neuron_C3[index] += sum;
// 				}
// 				neuron_C3[index] += bias_C3[channel];     //加偏置
// 				neuron_C3[index] = activation_function_tanh(neuron_C3[index]);  //激励函数
// 			}
// 		}
// 	}
// 	return true;
// }

// bool CNN::Forward_S4()
// {
// 	float scale_factor = 1.0 / (width_kernel_pooling_CNN * height_kernel_pooling_CNN);
// 	for (int i=0; i<num_map_S4_CNN; i++) {
// 		int block = width_image_C3_CNN * height_image_C3_CNN * i; //C3
// 		for (int y=0; y<height_image_S4_CNN; y++) {
// 			for (int x=0; x<width_image_S4_CNN; x++) {
// 				int rows = y * width_kernel_pooling_CNN;
// 				int cols = x * height_kernel_pooling_CNN;
// 				int index = (i*height_image_S4_CNN*width_image_S4_CNN) + y*width_image_S4_CNN + x; //S4 当前神经元

//                 neuron_S4[index] = 0.0;
// 				for (int m = 0; m < width_kernel_pooling_CNN; m++) {
// 					for (int n = 0; n < height_kernel_pooling_CNN; n++) {
//                         neuron_S4[index] += weight_S4[i] * neuron_C3[(rows + m) * width_image_C3_CNN + cols + n + block];
// 					}
// 				}
// 				//
// 				neuron_S4[index] *= scale_factor;
// 				neuron_S4[index] += bias_S4[i] ;
// 				neuron_S4[index] = activation_function_tanh(neuron_S4[index]);
// 			}
// 		}
// 	}
// 	return true;
// }

// bool CNN::Forward_C5()
// {
// #if 1
// 	for (int channel = 0; channel < num_map_C5_CNN; channel++) {
// 		for (int y = 0; y < height_image_C5_CNN; y++) {
// 			for (int x = 0; x < width_image_C5_CNN; x++) {
// 				int index = (channel*height_image_C5_CNN*width_image_C5_CNN) + y*width_image_C5_CNN + x;  //当前神经元
// 				neuron_C5[index] = 0.0;
// 				//卷积运算
// 				for (int inc = 0; inc < num_map_S4_CNN; inc++) {
// 					int addr1 = get_index(0, 0, num_map_S4_CNN * channel + inc, width_kernel_conv_CNN, height_kernel_conv_CNN, num_map_C5_CNN * num_map_S4_CNN);
// 					int addr2 = get_index(0, 0, inc, width_image_S4_CNN, height_image_S4_CNN, num_map_S4_CNN);
// 					const float* pw = &weight_C5[0] + addr1;       //卷积核
// 					const float* pi = &neuron_S4[0] + addr2;   //输入图像
// 					float sum = 0.0;
// 					const float* ppw = pw;
// 					const float* ppi = pi + y * width_image_S4_CNN + x;
// 					for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
// 						for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
// 							sum += *ppw++ * ppi[wy * width_image_S4_CNN + wx];
// 						}
// 					}
// 					neuron_C5[index] += sum;
// 				}
// 				neuron_C5[index] += bias_C5[channel];     //加偏置
// 				neuron_C5[index] = activation_function_tanh(neuron_C5[index]);  //激励函数
// 			}
// 		}
// 	}
// #else
// 	for (int channel = 0; channel < num_map_C5_CNN; channel++) {
// 		for (int y = 0; y < height_image_C5_CNN; y++) {
// 			for (int x = 0; x < width_image_C5_CNN; x++) {
// 				int index = (channel*height_image_C5_CNN*width_image_C5_CNN) + y*width_image_C5_CNN + x;  //C5 当前神经元
// 				for (int inc = 0; inc < num_map_S4_CNN; inc++) {
// 					int addr1 = width_kernel_conv_CNN*height_kernel_conv_CNN*(num_map_S4_CNN * channel + inc); //找到对应的卷积核
// 					int addr2 = height_image_S4_CNN*width_image_S4_CNN*inc;   //找到对应的S4输入
// 					addr2 += y * width_image_S4_CNN + x;
// 					//const float* pw = &weight_C5[0] + addr1;       //卷积核
// 					//const float* pi = &neuron_S4[0] + addr2;       //输入图像
// 					float sum = 0.0;
// 					//const float* ppw = pw;
// 					//const float* ppi = pi + y * width_image_S4_CNN + x;
// 					for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
// 						for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
//                             int addr3 = wy*width_kernel_conv_CNN + wx;  //卷积核索引
//                             int addr4 = wy*width_image_S4_CNN + wx;     //S4中的像素索引
//                             sum += weight_C5[addr1 + addr3]*neuron_S4[addr2+addr4];
// 							//sum += *ppw++ * ppi[wy * width_image_S4_CNN + wx];
// 						}
// 					}
// 					neuron_C5[index] += sum;
// 				}
// 				neuron_C5[index] += bias_C5[channel];     //加偏置
// 				neuron_C5[index] = activation_function_tanh(neuron_C5[index]);  //激励函数
// 			}
// 		}
// 	}
// #endif
// 	return true;
// }

// bool CNN::Forward_output()
// {
// 	for (int i = 0; i < num_neuron_output_CNN; i++) {
// 		neuron_output[i] = 0.0;
// 		for (int c = 0; c < num_neuron_C5_CNN; c++) {
// 			neuron_output[i] += weight_output[c * num_neuron_output_CNN + i] * neuron_C5[c];
// 		}
// 		neuron_output[i] += bias_output[i];
// 		neuron_output[i] = activation_function_tanh(neuron_output[i]);
// 	}
// 	return true;
// }






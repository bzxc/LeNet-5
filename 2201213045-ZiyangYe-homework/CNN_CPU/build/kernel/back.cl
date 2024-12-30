// 各层图像大小
#define width_image_input_CNN		32 //归一化图像宽
#define height_image_input_CNN		32 //归一化图像高
#define width_image_C1_CNN          28
#define height_image_C1_CNN		    28
#define width_image_S2_CNN		    14
#define height_image_S2_CNN		    14
#define width_image_C3_CNN		    10
#define height_image_C3_CNN		    10
#define width_image_S4_CNN		    5
#define height_image_S4_CNN		    5
#define width_image_C5_CNN		    1
#define height_image_C5_CNN		    1
#define width_image_output_CNN		1
#define height_image_output_CNN		1

// 卷积核大小
#define width_kernel_conv_CNN		5 //卷积核大小
#define height_kernel_conv_CNN		5
#define width_kernel_pooling_CNN	2
#define height_kernel_pooling_CNN	2
#define size_pooling_CNN		    2

// 特征图数量   feature maps
#define num_map_input_CNN		1 //输入层map个数
#define num_map_C1_CNN			6 //C1层map个数
#define num_map_S2_CNN			6 //S2层map个数
#define num_map_C3_CNN			16 //C3层map个数
#define num_map_S4_CNN			16 //S4层map个数
#define num_map_C5_CNN			120 //C5层map个数
#define num_map_output_CNN		10 //输出层map个数

// MNIST
#define num_patterns_train_CNN		60000  //60000 //训练模式对数(总数)
#define num_patterns_test_CNN		10000   //10000 //测试模式对数(总数)

// Train
#define num_epochs_CNN			    100   //最大迭代次数
#define accuracy_rate_CNN		    0.985 //要求达到的准确率
#define learning_rate_CNN		    0.01  //学习率
#define eps_CNN				        1e-8

//
#define len_weight_C1_CNN		150   //C1层权值数，5*5*6*1=150
#define len_bias_C1_CNN			6     //C1层阈值数，6
#define len_weight_S2_CNN		6     //S2层权值数,1*6=6
#define len_bias_S2_CNN			6     //S2层阈值数,6
#define len_weight_C3_CNN		2400  //C3层权值数，5*5*16*6=2400
#define len_bias_C3_CNN			16    //C3层阈值数,16
#define len_weight_S4_CNN		16    //S4层权值数，1*16=16
#define len_bias_S4_CNN			16    //S4层阈值数，16
#define len_weight_C5_CNN		48000 //C5层权值数，5*5*16*120=48000
#define len_bias_C5_CNN			120   //C5层阈值数，120
#define len_weight_output_CNN	1200  //输出层权值数，120*10=1200
#define len_bias_output_CNN		10    //输出层阈值数，10

#define num_neuron_input_CNN     1024 //输入层神经元数，32*32=1024
#define num_neuron_C1_CNN        4704 //C1层神经元数，28*28*6=4704
#define num_neuron_S2_CNN		 1176 //S2层神经元数，14*14*6=1176
#define num_neuron_C3_CNN		 1600 //C3层神经元数，10*10*16=1600
#define num_neuron_S4_CNN		 400  //S4层神经元数，5*5*16=400
#define num_neuron_C5_CNN		 120  //C5层神经元数，1*120=120
#define num_neuron_output_CNN    10   //输出层神经元数，1*10=10


//globalid  10
__kernel void kernel_backward_output( __global float* neuron_output, __global float * data_single_label, __global float* delta_neuron_output, int index) 
{
    const int id = get_global_id(0);
    int n = neuron_output[id];
    int d = data_single_label[id];
    int de_dy = n - d;
    int dy_da = 1 - n*n;
    delta_neuron_output[id] = de_dy * dy_da;
}


//globalid 120
__kernel void kernel_backward_C5_OW(__global float* delta_neuron_C5, __global float* delta_weight_output,  __global float* delta_bias_output,
                                    __global float * delta_neuron_output, __global float * weight_output, __global float * neuron_C5)
{
    // const int j = get_global_id(0);  不然内存一致性无法保证
    const int k = get_global_id(0);
    for(int j = 0; j < 10; j++){
        int addr1 = k * num_neuron_output_CNN + j;
        int addr2 = j;
        int temp = 1 - neuron_C5[k] * neuron_C5[k];
        delta_neuron_C5[k] += delta_neuron_output[j] * weight_output[addr1] * temp;
        delta_weight_output[addr1] += delta_neuron_output[j] * neuron_C5[k]; //这里+=和=应该是等价的
        // delta_bias_output[addr2] 可能要单独更新了
    }
    if(k < 10) {
        delta_bias_output[j] = num_neuron_C5_CNN * delta_neuron_output[j];
    }
}
// //g 10
// __kernel void kernel_backward_C5_b( __global float* delta_bias_output, __global float * delta_neuron_output) 
// {
//     int j = get_global_id(0);
//     delta_bias_output[j] = num_neuron_C5_CNN * delta_neuron_output[j];
// }

//g 120 16
__kernel void kernel_backward_S4_OW(__global float* delta_neuron_S4, __global float* delta_weight_C5,
                                    __global float* delta_neuron_C5, __global float* weight_C5,
                                    __global float* neuron_S4) 
{
    int outc = get_global_id(0);
    int index = outc;
    int inc = get_global_id(1);
    int addr1 = width_kernel_conv_CNN*height_kernel_conv_CNN*(num_map_S4_CNN * outc + inc);
    int addr2 = height_image_S4_CNN*width_image_S4_CNN*inc;
    float ttemp1 = 0;
    float ttemp2 = 0;
    for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
        for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
            int addr3 = addr1 + wy*width_kernel_conv_CNN + wx;  //卷积核索引 W_kj
            int addr4 = addr2 + wy*width_image_S4_CNN + wx;     //S4中的像素索引 S4 k
            int addr5 = outc;
            float temp = 1 - neuron_S4[addr4] * neuron_S4[addr4];
            if(outc == 0) delta_neuron_S4[addr4] = delta_neuron_C5[index] * weight_C5[addr3]
                                    * temp;
            delta_weight_C5[addr3] = delta_neuron_C5[index] * neuron_S4[addr4];
        }
    }
}
//g 120
__kernel void kernel_backward_S4_b( __global float* delta_bias_C5, __global float * delta_neuron_C5)
{
    int i = get_global_id(0);
    delta_bias_C5[i] = 25 * 16 * delta_neuron_C5[i];
}

//g 16 5 5
__kernel void kernel_backward_C3_O(__global float* delta_neuron_C3,
                                    __global float* delta_neuron_S4, __global float* weight_S4,
                                    __global float* neuron_C3)
{
    int outc = get_global_id(0);
    int block = width_image_C3_CNN * height_image_C3_CNN * outc; //C3
    int y = get_global_id(1);
    int x = get_global_id(2);
    float scale_factor = 1.0 /(height_kernel_pooling_CNN * width_kernel_pooling_CNN);
    int rows = y * width_kernel_pooling_CNN;
    int cols = x * height_kernel_pooling_CNN;
    int index = (outc*height_image_S4_CNN*width_image_S4_CNN) + y*width_image_S4_CNN + x;
    for (int m = 0; m < height_kernel_pooling_CNN; m++) {
        for (int n = 0; n < width_kernel_pooling_CNN; n++) {
            int addr1 = outc;  // 权重
            int addr2 = block + (rows + m) * width_image_C3_CNN + cols + n; //C3 神经元 k
            // int addr3 = outc;
            int temp = 1 - neuron_C3[addr2] * neuron_C3[addr2];
            delta_neuron_C3[addr2] += delta_neuron_S4[index] * weight_S4[addr1]
                                    * temp * scale_factor;
            // delta_weight_S4[addr1] += delta_neuron_S4[index] * neuron_C3[addr2] * scale_factor;
        }
    }
}

//g 16
__kernel void kernel_backward_C3_Wb(__global float* delta_weight_S4, __global float* delta_bias_S4, 
                                    __global float * delta_neuron_S4,
                                    __global float * neuron_C3)
{
    int outc = get_global_id(0);
    float scale_factor = 1.0 / (width_kernel_pooling_CNN * height_kernel_pooling_CNN);
    int block = width_image_C3_CNN * height_image_C3_CNN * outc; //C3
    for (int y=0; y<height_image_S4_CNN; y++) {
        for (int x=0; x<width_image_S4_CNN; x++) {
            int rows = y * width_kernel_pooling_CNN;
            int cols = x * height_kernel_pooling_CNN;
            int index = (outc*height_image_S4_CNN*width_image_S4_CNN) + y*width_image_S4_CNN + x; //S4 当前神经元j

            for (int m = 0; m < height_kernel_pooling_CNN; m++) {
                for (int n = 0; n < width_kernel_pooling_CNN; n++) {
                    int addr1 = outc;  // 权重
                    int addr2 = block + (rows + m) * width_image_C3_CNN + cols + n; //C3 神经元 k
                    int addr3 = outc;
                    delta_weight_S4[addr1] += delta_neuron_S4[index] * neuron_C3[addr2] * scale_factor;
                    delta_bias_S4[addr3] += delta_neuron_S4[index];
                }
            }
        }//index
    }
    delta_bias_S4[outc] = delta_neuron_S4[outc] * 5 * 5 * 2 * 2;
}

//g 10 10 6
__kernel void kernel_backward_S2_O(__global float* delta_neuron_S2, 
                                    __global float* delta_neuron_C3, __global float* weight_C3, 
                                    __global float* neuron_S2)
{
    const int tbl[6][16] = {
        0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0,
        1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0,
        1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0,
        1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0
    };
    int y = get_global_id(0);
    int x = get_global_id(1);
    int inc = get_global_id(2);
    for(int outc = 0; outc < num_map_C3_CNN; outc++) {
        int index = (outc*height_image_C3_CNN*width_image_C3_CNN) + y*width_image_C3_CNN + x;
        if(tbl[inc][outc] == 1) {
            int addr1 = width_kernel_conv_CNN*height_kernel_conv_CNN*(num_map_S2_CNN * outc + inc); //找到对应的卷积核
            int addr2 = height_image_S2_CNN*width_image_S2_CNN*inc;   //找到对应的S2输入
            addr2 +=  y * width_image_S2_CNN + x;  //S2 k

            for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
                for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
                    int addr3 = addr1 + wy*width_kernel_conv_CNN + wx;  //卷积核索引 W_kj
                    int addr4 = addr2 + wy*width_image_S2_CNN + wx;     //S2中的像素索引 S2 k
                    int addr5 = outc;
                    int temp = 1 - neuron_S2[addr4] * neuron_S2[addr4];
                    delta_neuron_S2[addr4] += delta_neuron_C3[index] * weight_C3[addr3]
                                            * temp;
                }
            }
        }
    }

}


//g 16 10 10
__kernel void kernel_backward_S2_W(__global float* delta_weight_C3, 
                                    __global float* delta_neuron_C3, __global float* neuron_S2)
{
    const int tbl[6][16] = {
        0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0,
        1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0,
        1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0,
        1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0
    };
    int outc = get_global_id(0);
    int y = get_global_id(1);
    int x = get_global_id(2);
    int index = (outc*height_image_C3_CNN*width_image_C3_CNN) + y*width_image_C3_CNN + x;  //C3 当前神经元 j
    for (int inc = 0; inc < num_map_S2_CNN; inc++) {
        if (!tbl[inc][outc]) continue;
        int addr1 = width_kernel_conv_CNN*height_kernel_conv_CNN*(num_map_S2_CNN * outc + inc); //找到对应的卷积核
        int addr2 = height_image_S2_CNN*width_image_S2_CNN*inc;   //找到对应的S2输入
        addr2 +=  y * width_image_S2_CNN + x;  //S2 k

        for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
            for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
                int addr3 = addr1 + wy*width_kernel_conv_CNN + wx;  //卷积核索引 W_kj
                int addr4 = addr2 + wy*width_image_S2_CNN + wx;     //S2中的像素索引 S2 k
                int addr5 = outc;
                delta_weight_C3[addr3] += delta_neuron_C3[index] * neuron_S2[addr4];
            }
        }

    }
}

//g 16
__kernel void kernel_backward_S2_b( __global float* delta_bias_C3, __global float * delta_neuron_C3) 
{
    int outc = get_global_id(0);
    for (int y = 0; y < height_image_C3_CNN; y++) {
			for (int x = 0; x < width_image_C3_CNN; x++) {
				int index = (outc*height_image_C3_CNN*width_image_C3_CNN) + y*width_image_C3_CNN + x;
                delta_bias_C3[outc] += delta_neuron_C3[index] * 6 * 5 * 5;
            }
    }
}

//6 14 14
__kernel void kernel_backward_C1_O(__global float* delta_neuron_C1,
                                    __global float* delta_neuron_S2, __global float* weight_S2,
                                    __global float* neuron_C1)
{
    int outc = get_global_id(0);
    float scale_factor = 1.0 / (width_kernel_pooling_CNN * height_kernel_pooling_CNN);
    int block = width_image_C1_CNN * height_image_C1_CNN * outc; //C1
    int y = get_global_id(1);
    int x = get_global_id(2);
    int rows = y * width_kernel_pooling_CNN;
    int cols = x * height_kernel_pooling_CNN;
    int index = (outc*height_image_S2_CNN*width_image_S2_CNN) + y*width_image_S2_CNN + x; //S2 当前神经元j

    for (int m = 0; m < height_kernel_pooling_CNN; m++) {
        for (int n = 0; n < width_kernel_pooling_CNN; n++) {
            int addr1 = outc;  // 权重
            int addr2 = block + (rows + m) * width_image_C1_CNN + cols + n; //C1 神经元 k
            int addr3 = outc;
            int temp = 1 - neuron_C1[addr2] * neuron_C1[addr2];
            delta_neuron_C1[addr2] += delta_neuron_S2[index] * weight_S2[addr1]
                                    * temp * scale_factor;
        }
    }

}

//g 6
__kernel void kernel_backward_C1_Wb(__global float* delta_weight_S2, __global float* delta_bias_S2, 
                                    __global float * delta_neuron_S2,
                                    __global float * neuron_C1)
{
    int outc = get_global_id(0);
    float scale_factor = 1.0 / (width_kernel_pooling_CNN * height_kernel_pooling_CNN);
    int block = width_image_C1_CNN * height_image_C1_CNN * outc; //C1
    for (int y=0; y<height_image_S2_CNN; y++) {
        for (int x=0; x<width_image_S2_CNN; x++) {
            int rows = y * width_kernel_pooling_CNN;
            int cols = x * height_kernel_pooling_CNN;
            int index = (outc*height_image_S2_CNN*width_image_S2_CNN) + y*width_image_S2_CNN + x; //S2 当前神经元j

            for (int m = 0; m < height_kernel_pooling_CNN; m++) {
                for (int n = 0; n < width_kernel_pooling_CNN; n++) {
                    int addr1 = outc;  // 权重
                    int addr2 = block + (rows + m) * width_image_C1_CNN + cols + n; //C1 神经元 k
                    int addr3 = outc;
                    delta_weight_S2[addr1] += delta_neuron_S2[index] * neuron_C1[addr2] * scale_factor;
                    delta_bias_S2[addr3] += delta_neuron_S2[index];
                }
            }
        }//index
    }
}

//g 28 28
__kernel void kernel_backward_input_O(__global float* delta_neuron_input, 
                                    __global float* delta_neuron_C1, __global float* weight_C1, 
                                    __global float* data_single_image)
{
    int y = get_global_id(0);
    int x = get_global_id(1);
    int inc = 0;
    for (int outc = 0; outc < num_map_C1_CNN; outc++) {
        int index = (outc*height_image_C1_CNN*width_image_C1_CNN) + y*width_image_C1_CNN + x;
        int addr1 = width_kernel_conv_CNN*height_kernel_conv_CNN*(num_map_input_CNN * outc + inc); //找到对应的卷积核
        int addr2 = height_image_input_CNN*width_image_input_CNN*inc;   //找到对应的input输入 0
        addr2 +=  y * width_image_input_CNN + x;  //input k

        for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
            for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
                int addr3 = addr1 + wy*width_kernel_conv_CNN + wx;  //卷积核索引 W_kj
                int addr4 = addr2 + wy*width_image_input_CNN + wx;     //input中的像素索引 input k
                int addr5 = outc;
                int temp = 1 - data_single_image[addr4] * data_single_image[addr4];
                delta_neuron_input[addr4] += delta_neuron_C1[index] * weight_C1[addr3]
                                        * temp;
            }
        }
    }
}

//g 6 28 28
__kernel void kernel_backward_input_W(__global float* delta_weight_C1, 
                                    __global float* delta_neuron_C1, __global float* data_single_image)
{
    int outc = get_global_id(0);
    int y = get_global_id(1);
    int x = get_global_id(2);
    int index = (outc*height_image_C1_CNN*width_image_C1_CNN) + y*width_image_C1_CNN + x;  //C1 当前神经元 j
    for (int inc = 0; inc < num_map_input_CNN; inc++) {
        int addr1 = width_kernel_conv_CNN*height_kernel_conv_CNN*(num_map_input_CNN * outc + inc); //找到对应的卷积核
        int addr2 = height_image_input_CNN*width_image_input_CNN*inc;   //找到对应的input输入 0
        addr2 +=  y * width_image_input_CNN + x;  //input k

        for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
            for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
                int addr3 = addr1 + wy*width_kernel_conv_CNN + wx;  //卷积核索引 W_kj
                int addr4 = addr2 + wy*width_image_input_CNN + wx;     //input中的像素索引 input k
                delta_weight_C1[addr3] += delta_neuron_C1[index] * data_single_image[addr4];
            }
        }
    }
}

//g 6
__kernel void kernel_backward_input_b( __global float* delta_bias_C1, __global float * delta_neuron_C1) 
{
    int outc = get_global_id(0);
    for (int y = 0; y < height_image_C1_CNN; y++) {
			for (int x = 0; x < width_image_C1_CNN; x++) {
				int index = (outc*height_image_C1_CNN*width_image_C1_CNN) + y*width_image_C1_CNN + x;
                delta_bias_C1[outc] += delta_neuron_C1[index] * 6 * 5 * 5;
            }
    }
}
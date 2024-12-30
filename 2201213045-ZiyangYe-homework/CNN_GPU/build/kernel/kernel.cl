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
#define BS1 7
#define BS2 14
#define BS3 5
#define BS4 5
#define BS5 1
#define filtersize 5
#define convsize 2
#define BX 2
#define BY 1

__constant int tbl[6][16] = {
	{1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1},
	{1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1},
	{1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1},
	{0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1},
	{0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1},
	{0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1}
};

//--------------------------------------------------------
//         local + constant
//--------------------------------------------------------
// __kernel void  kernel_forward_c1(__global float *in,//data_single_image->data_input_train
//                       __constant float  *weight,//weight_C1 卷积核
//                       __constant float  *bias,//每个特征图有自己的bias_C1[len_bias_C1_CNN]
//                       __global float  *out,//neuron_C1[]
// 					  int input_index 
//                     //   int channel,//num_map_C1_CNN 特征图的数量 6 
//                     //   int out_width,//width_image_c1_CNN 28
//                     //   int out_height,//height_image_c1_CNN 28
//                     //   int kernel_width,//卷积核 5
// 					//   int kernel_height,//卷积核 5
// 					//   int in_num,//num_map_input_CNN 特征图的数量
// 					//   int in_width,//width_image_input_CNN 32 
//                     //   int in_height//height_image_input_CNN 32
// 					  )
// {//每个神经元节点输出是并行的
// 	int channel = get_global_id(0);
// 	int out_height = 28;
// 	int out_width = 28;
//     int  y = get_global_id(1);
//     int  x = get_global_id(2);
// 	int kernel_width = 5;
// 	// printf("0:%d %d %d\n", channel, y, x);
// 	int kernel_height = 5;
// 	int in_width = 32;
// 	int in_height = 32;
// 	int in_num = 1;

//         int tidy=get_local_id(1);//[0,BY1)
//         int tidx=get_local_id(2);//[0,BX1)
//         float local pixel[num_map_input_CNN][BS1+filtersize-1][BS1+filtersize-1];


//         for (int i=0; i<in_num; i++)
//         {
//                 int addr2 = i*in_width*in_height;
//                 pixel[i][tidy][tidx]=in[addr2 + y*in_width + x];
// 				if(tidx < filtersize -1 ) pixel[i][tidy][tidx+BS1] = in[addr2+y*in_width+x+BS1];
//                if(tidy < filtersize -1 ) pixel[i][tidy+BS1][tidx] = in[addr2+(y+BS1)*in_width+x];
// 			   if(tidx < filtersize -1 &&tidy < filtersize-1) 
// 			   				pixel[i][tidy+BS1][tidx+BS1] = in[addr2+(y+BS1)*in_width+x+BS1];
//         }
//         barrier(CLK_LOCAL_MEM_FENCE);
		
//         int  index = (channel*out_height*out_width) + y*out_width + x;
//         float sum = 0.0;
// 		int inc = 0;
// 		int wx = 0;
// 		int wy = 0;
// 		out[index] = 0.0;
//         for (inc=0; inc<in_num; inc++)
//         {
// 		int addr1 = (in_num * channel + inc) * kernel_height * kernel_width;
// 		int addr2 = inc*(BS1+filtersize-1)*(BS1+filtersize-1);
// 		sum = 0.0;
// 		__constant const float* pw = weight + addr1;   //卷积核,默认是__private变量
// 		__local const float* pi = &pixel[0][0][0] + addr2;       //输入图像
// 		__constant const float* ppw = pw;//卷积核
// 		__local const float* ppi = pi + tidy * (BS1+filtersize-1) + tidx;//输入图像

		
//         for(wy = 0; wy < kernel_height; wy++)
//                 {
//                         for(wx = 0; wx < kernel_width; wx++)
//                         {
//                 sum += *ppw++ *ppi[wy*(BS1+filtersize-1)+wx]  ;//pixel[inc][tidy+wy][tidx+wx];
//                     }
//             }
// 			out[index] += sum;
        
// 		}
//         out[index] += bias[channel];
//         out[index] = tanh((float)out[index]);
		
// }
__kernel void  kernel_forward_s2(__global float *in,//neuron_C1
					  constant float  *weight,//weight_S2[] 
                      constant float  *bias,//每个特征图有自己的bias_C2[len_bias_C2_CNN]
                      __global float  *out//neuron_S2[] 
                    //   int channel,//num_map_C2_CNN 特征图的数量 6 
                    //   int out_width,//width_image_C2_CNN 14
                    //   int out_height,//height_image_C2_CNN 14
                    //   int kernel_width, //池化核 2
					//   int kernel_height,//池化核 2
					//   int in_num,//num_map_C1_CNN 特征图的数量 6
					//   int in_width,//width_image_C1_CNN 28
                    //   int in_height//height_image_C1_CNN 28
					  )
{
	int channel = get_global_id(0);
    int  y = get_global_id(1);
    int  x = get_global_id(2);
	//local mem
	int tidy=get_local_id(1);
    int tidx=get_local_id(2);
	int out_height = 14;
	int out_width = 14;
	int kernel_width=2;
	int kernel_height=2;
	int in_width=28;
	int in_height=28;
	float local pixel[num_map_C1_CNN][BS2<<1][BS2<<1];
	pixel[channel][tidy][tidx] = in[channel*in_width*in_height+y*in_width+x];
	pixel[channel][tidy+BS2][tidx] = in[channel*in_width*in_height+(y+BS2)*in_width+x];
	pixel[channel][tidy][tidx+BS2] = in[channel*in_width*in_height+y*in_width+x+BS2];
	pixel[channel][tidy+BS2][tidx+BS2] = in[channel*in_width*in_height+(y+BS2)*in_width+x+BS2];
	barrier(CLK_LOCAL_MEM_FENCE);
	//
    //float scale_factor = 1.0 / (kernel_width * kernel_height);
    //int block =	 channel;
    int rows = tidy * kernel_width;
	int cols = tidx * kernel_height;
	int index = (channel*out_height*out_width) + y*out_width + x;

	out[index] = 0.0;
	for (int m = 0; m < kernel_width; m++) {
		for (int n = 0; n < kernel_height; n++) {
            out[index] += weight[channel] * pixel[channel][rows + m][cols + n];
		}
	}
	out[index] *= 0.25;  //scale_factor 池化层是2*2的 1/4;
	out[index] += bias[channel] ;
	out[index] = tanh((float)(out[index]));
}

__kernel void  kernel_forward_c3(__global float *in,//neuron_S2[]
                      constant float  *weight,//weight_C3[]
                      constant float  *bias,//bias_C3[]
                      __global float  *out//neuron_C3[]
                    //   int channel,//num_map_C3_CNN 特征图的数量 16 
                    //   int out_width,//width_image_c3_CNN 10
                    //   int out_height,//height_image_c3_CNN 10
                    //   int kernel_width,//卷积核 5
					//   int kernel_height,//卷积核 5
					//   int in_num,//num_map_S2_CNN 特征图的数量6
					//   int in_width,//width_image_S2_CNN 14 
                    //   int in_height,//height_image_S2_CNN 14
                    //   __global bool  *tbl //bool tbl[6][16]
					  )
{
	int channel = get_global_id(0);
	int out_height = 10;
	int out_width = 10;
    int y = get_global_id(1);
    int x = get_global_id(2);
	int kernel_width = 5;
	int kernel_height = 5;
	int in_width = 14;
	int in_height = 14;
	int in_num = 6;
	//local mem
	int tidy=get_local_id(1);
    int tidx=get_local_id(2);
	float local pixel[num_map_S2_CNN][BS3+filtersize-1][BS3+filtersize-1];
	for (int i=0; i<in_num; i++){
        int addr2 = i*in_width*in_height;
        pixel[i][tidy][tidx]=in[addr2 + y*in_width + x];
		if(tidx < filtersize -1 ) pixel[i][tidy][tidx+BS3] = in[addr2+y*in_width+x+BS3];
        if(tidy < filtersize -1 ) pixel[i][tidy+BS3][tidx] = in[addr2+(y+BS3)*in_width+x];
		if(tidx < filtersize -1 &&tidy < filtersize-1) 
			   				pixel[i][tidy+BS3][tidx+BS3] = in[addr2+(y+BS3)*in_width+x+BS3];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
	//
    int  index = (channel*out_height*out_width) + y*out_width + x;
	float sum = 0.0;
	int inc = 0;
	int wx = 0;
	int wy = 0;
	out[index] = 0.0;
	for (inc=0; inc<in_num; inc++) {
		if (!tbl[inc*16+channel]) continue;
        int addr1 = (in_num * channel + inc) * kernel_height * kernel_width;
		int addr2 = (inc)*(BS3+filtersize-1)*(BS3+filtersize-1);
		__constant const float* pw = weight + addr1;   //卷积核
		__local const float* pi = &pixel[0][0][0] + addr2;       //输入图像
		sum = 0.0;
		__constant const float* ppw = pw;
		__local const float* ppi = pi + tidy * (BS3+filtersize-1) + tidx;
        for(wy = 0; wy < kernel_height; wy++)  {
			for(wx = 0; wx < kernel_width; wx++) {
                sum += *ppw++ * ppi[wy * (BS3+filtersize-1)+ wx];
		    }
	     }
	     out[index] += sum;
	}
	out[index] += bias[channel];
	out[index] = tanh((float)(out[index]));
}

__kernel void  kernel_forward_s4(__global float *in,
                      constant float  *weight,
                      constant float  *bias,
                      __global float  *out
                    //   int channel,
                    //   int out_width,
                    //   int out_height,
                    //   int kernel_width,
					//   int kernel_height,
					//   int in_num,
					//   int in_width,
                    //   int in_height
					  )
{
	int channel = get_global_id(0);
	int out_height = 5;
	int out_width = 5;
    int  y = get_global_id(1);
    int  x = get_global_id(2);
	int kernel_width=2;
	int kernel_height=2;
	int in_width=10;
	int in_height=10;
	//local mem
	int tidy=get_local_id(1);
    int tidx=get_local_id(2);
	float local pixel[num_map_C3_CNN][BS4<<1][BS4<<1];
	pixel[channel][tidy][tidx] = in[channel*in_width*in_height+y*in_width+x];
	pixel[channel][tidy+BS4][tidx] = in[channel*in_width*in_height+(y+BS4)*in_width+x];
	pixel[channel][tidy][tidx+BS4] = in[channel*in_width*in_height+y*in_width+x+BS4];
	pixel[channel][tidy+BS4][tidx+BS4] = in[channel*in_width*in_height+(y+BS4)*in_width+x+BS4];
	barrier(CLK_LOCAL_MEM_FENCE);
    //float scale_factor = 1.0 / (kernel_width * kernel_height);
    //int block = in_width * in_height * channel;
    int rows = tidy * kernel_width;
	int cols = tidx * kernel_height;
	int index = (channel*out_height*out_width) + y*out_width + x;

	out[index] = 0.0;
	for (int m = 0; m < kernel_width; m++) {
		for (int n = 0; n < kernel_height; n++) {
            out[index] += weight[channel] * pixel[channel][rows + m][ cols + n];
		}
	}
	out[index] *= 0.25;  //scale_factor;
	out[index] += bias[channel] ;
	out[index] = tanh((float)(out[index]));
}

__kernel void  kernel_forward_c5(__global float *in,
                      __global float  *weight,//constant memory is 64KB
                      constant float  *bias,
                      __global float  *out
					  )
{
	int channel = get_global_id(0);
    int  y = get_global_id(1);
    int  x = get_global_id(2);
	int out_height=1;
	int out_width=1;
	int kernel_width = 5;
	int kernel_height = 5;
	int in_width = 5;
	int in_height = 5;
	int in_num=16;
	//local mem
	int tidy=get_local_id(1);
    int tidx=get_local_id(2);
	float __local pixel[num_map_S4_CNN][BS5+filtersize-1][BS5+filtersize-1];
	for (int i=0; i<in_num; i++){//16
        int addr2 = i*in_width*in_height;
        for(int j = 0;j < 5;++ j){
			for(int k = 0;k < 5;++ k){
				pixel[i][tidy+j][tidx+k] = in[addr2 + (y+j)*in_width + x + k];
			}
		}
	}
    barrier(CLK_LOCAL_MEM_FENCE);
	//
    int  index = (channel*out_height*out_width) + y*out_width + x;
	float sum = 0.0;
	int inc = 0;
	int wx = 0;
	int wy = 0;
	out[index] = 0.0;
	for (inc=0; inc<in_num; inc++) {
        int addr1 = (in_num * channel + inc) * kernel_height * kernel_width;
		int addr2 = (inc)*(BS5+filtersize-1)*(BS5+filtersize-1);
		__global const float* pw = weight + addr1;   //卷积核
		__local const float* pi = &pixel[0][0][0] + addr2;       //输入图像
		sum = 0.0;
		__global const float* ppw = pw;
		__local const float* ppi = pi + tidy * (BS5+filtersize-1) + tidx;
        for(wy = 0; wy < kernel_height; wy++)  {
			for(wx = 0; wx < kernel_width; wx++) {
                sum += *ppw++ * ppi[wy * (BS5+filtersize-1) + wx];
		    }
	     }
	     out[index] += sum;
	}
	out[index] += bias[channel];
	out[index] = tanh((float)(out[index]));
}

//globalid  10
__kernel void kernel_backward_output( __global float* neuron_output, __global float * data_single_label, __global float* delta_neuron_output, int index) 
{
    const int id = get_global_id(0);
    float n = neuron_output[id];
	__global float *labels = data_single_label + index;
    float d = labels[id];
    float de_dy = n - d;
    float dy_da = 1 - n*n;
    delta_neuron_output[id] = de_dy * dy_da;
}

//globalid 120
__kernel void kernel_backward_c5(__global float * delta_neuron_output, __global float * neuron_C5, __global float * weight_output, __global float* delta_weight_output, 
									__global float* delta_bias_output,  __global float* delta_neuron_C5
                                    )
{
    // const int j = get_global_id(0);  不然内存一致性无法保证
    const int k = get_global_id(0);
	float ttemp = 0;
    for(int j = 0; j < 10; j++){
        int addr1 = k * num_neuron_output_CNN + j;
        int addr2 = j;
        float temp = 1 - neuron_C5[k] * neuron_C5[k];
        ttemp += delta_neuron_output[j] * weight_output[addr1] * temp;
        delta_weight_output[addr1] += delta_neuron_output[j] * neuron_C5[k]; //这里+=和=应该是等价的
        // delta_bias_output[addr2] 可能要单独更新了
    }
	delta_neuron_C5[k] = ttemp;
    if(k < 10) {
        delta_bias_output[k] = num_neuron_C5_CNN * delta_neuron_output[k];
    }
}

//g 16 5 5
__kernel void kernel_backward_s4(__global float* delta_neuron_C5, __global float* neuron_S4,
                                    __global float* weight_C5, __global float* delta_weight_C5, __global float* delta_bias_C5,
                                    __global float* delta_neuron_S4) 
{
    int inc = get_global_id(0);
	int wy = get_global_id(1);
	int wx = get_global_id(2);
    int addr2 = height_image_S4_CNN*width_image_S4_CNN*inc;   //找到对应的S4输入
	int addr4 = addr2 + wy*width_image_S4_CNN + wx;     //S4中的像素索引 S4 k
	float out_addr4=0;
	float neuron_S4_addr4 = neuron_S4[addr4];
	for (int outc = 0; outc < num_map_C5_CNN; outc++) {
		int addr1 = width_kernel_conv_CNN*height_kernel_conv_CNN*(num_map_S4_CNN * outc + inc); //找到对应的卷积核

		int addr3 = addr1 + wy*width_kernel_conv_CNN + wx;  //卷积核索引 W_kj
		out_addr4 += delta_neuron_C5[outc] * weight_C5[addr3] * (1.0 - neuron_S4_addr4 * neuron_S4_addr4);
		delta_weight_C5[addr3] = delta_neuron_C5[outc] * neuron_S4_addr4;
		// delta_bias[outc] += in[outc];
		if(inc == 0 && wx == 0 && wy == 0)
			delta_bias_C5[outc] = delta_neuron_C5[outc]*400;
	}
	delta_neuron_S4[addr4] = out_addr4;
}



// g 16 5 5
__kernel void kernel_backward_c3(__global float* delta_neuron_S4, __global float* neuron_C3, __global float* weight_S4, 
								 __global float* delta_weight_S4,  __global float* delta_bias_S4, 
								 __global float* delta_neuron_C3
                                )
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
            float temp = 1.0 - neuron_C3[addr2] * neuron_C3[addr2];
            delta_neuron_C3[addr2] = delta_neuron_S4[index] * weight_S4[addr1]
                                    * temp * scale_factor;
            // delta_weight_S4[addr1] += delta_neuron_S4[index] * neuron_C3[addr2] * scale_factor;
        }
    }
	if(x==0&&y==0) {
		float tmpw = 0;
		float tmpb = 0;
		for (int yy=0; y<height_image_S4_CNN; y++) {
			for (int xx=0; x<width_image_S4_CNN; x++) {
				int rows = yy * width_kernel_pooling_CNN;
				int cols = xx * height_kernel_pooling_CNN;
				int index = (outc*height_image_S4_CNN*width_image_S4_CNN) + yy*width_image_S4_CNN + xx; //S4 当前神经元j
				
				for (int m = 0; m < height_kernel_pooling_CNN; m++) {
					for (int n = 0; n < width_kernel_pooling_CNN; n++) {
						int addr1 = outc;  // 权重
						int addr2 = block + (rows + m) * width_image_C3_CNN + cols + n; //C3 神经元 k
						int addr3 = outc;
						tmpw += delta_neuron_S4[index] * neuron_C3[addr2] * scale_factor;
						tmpb += delta_neuron_S4[index];
					}
				}
			}//index
		}
		delta_weight_S4[outc] = tmpw;
		delta_bias_S4[outc] = tmpb;
	}
}


__kernel void  kernel_backward_s2(__global float *delta_neuron_C3, __global float *neuron_S2, __global float *weight_C3,
									__global float *delta_neuron_S2
){
	//[14,14,6]
	int yy = get_global_id(0);
	int xx = get_global_id(1);
	int inc = get_global_id(2);
	int addr4 = 14*14*inc+yy*14+xx;
	float out_addr4 = 0;
	float neuron_S2_addr4 = neuron_S2[addr4];
	for (int outc = 0; outc < 16; outc++) {
		if (!tbl[inc][outc]) continue;
		int addr1 = 5*5*(6 * outc + inc); //找到对应的卷积核
		for(int y = max(yy-4,0);y<=min(yy,9);y++){
			int wy = yy - y;
			for(int x = max(xx-4,0);x<=min(xx,9);x++){
				int wx = xx - x;
				int index = (outc*10*10) + y*10 + x;  //C3 当前神经元 j
				int addr3 = addr1 + wy*5 + wx;  //卷积核索引 W_kj
				out_addr4 += delta_neuron_C3[index] * weight_C3[addr3] * (1-neuron_S2_addr4*neuron_S2_addr4);
			}
		}
	}
	delta_neuron_S2[addr4] = out_addr4;
}

__kernel void  kernel_backward_s2_weight(
	__global float *delta_neuron_C3, 
	__global float *neuron_S2, 
	__global float *delta_weight_C3 
){
	//[25,16,6]
	int wxy = get_global_id(0);
	int outc = get_global_id(1);
	int inc = get_global_id(2);
	int wy = wxy / 5;
	int wx = wxy - wy * 5;
	if (!tbl[inc][outc]) 
		return;
	int addr3 = 5*5*(6 * outc + inc) + wxy;
	float delta_weight_addr3 = 0;

	for (int y = 0; y < 10; y++) {
		for (int x = 0; x < 10; x++) {
			int index = (outc*10*10) + y*10 + x;  //C3 当前神经元 j
			int addr2 = 14*14*inc +  y * 14 + x + wy*14 + wx;   //找到对应的S2输入
			delta_weight_addr3 += delta_neuron_C3[index] * neuron_S2[addr2];
		}
	}
	delta_weight_C3[addr3] = delta_weight_addr3;
}

__kernel void  kernel_backward_s2_bias(__global float *delta_neuron_C3, __global float *delta_bias_C3
){
	//[16]
	int outc = get_global_id(0);
	float delta_bias_outc = 0;

	for (int inc = 0; inc < 6; inc++) {
		if (!tbl[inc][outc]) continue;
		for (int y = 0; y < 10; y++) {
			for (int x = 0; x < 10; x++) {
				int index = (outc*10*10) + y*10 + x;  //C3 当前神经元 j
				delta_bias_outc += delta_neuron_C3[index]*25;
			}
		}
	}
	delta_bias_C3[outc] = delta_bias_outc;
}

__kernel void kernel_backward_c1(__global float* delta_neuron_S2, __global float* neuron_C1, __global float* weight_S2, 
									__global float* delta_weight_S2, __global float* delta_bias_S2, __global float* delta_neuron_C1 
                                    )
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
            delta_neuron_C1[addr2] = delta_neuron_S2[index] * weight_S2[addr1]
                                    * temp * scale_factor;
        }
    }

	if(x==0 && y == 0) {
		float scale_factor = 1.0 / (width_kernel_pooling_CNN * height_kernel_pooling_CNN);
		int block = width_image_C1_CNN * height_image_C1_CNN * outc; //C1
		float tmpw = 0;
		float tmpb = 0;
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
						tmpw += delta_neuron_S2[index] * neuron_C1[addr2] * scale_factor;
						tmpb += delta_neuron_S2[index];
					}
				}
			}//index
    	}
		delta_weight_S2[outc] = tmpw;
		delta_bias_S2[outc] = tmpb;
	}

}

//g 6 28 28
__kernel void kernel_backward_input_weight(__global float* delta_neuron_C1, __global float* data_single_image, 
										__global float* delta_weight_C1,
										int index
                                     )
{
    int outc = get_global_id(0);
    int y = get_global_id(1);
    int x = get_global_id(2);
    index = (outc*height_image_C1_CNN*width_image_C1_CNN) + y*width_image_C1_CNN + x;  //C1 当前神经元 j
    for (int inc = 0; inc < num_map_input_CNN; inc++) {
        int addr1 = width_kernel_conv_CNN*height_kernel_conv_CNN*(num_map_input_CNN * outc + inc); //找到对应的卷积核
        int addr2 = height_image_input_CNN*width_image_input_CNN*inc;   //找到对应的input输入 0
        addr2 +=  y * width_image_input_CNN + x;  //input k

        for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
            for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
                int addr3 = addr1 + wy*width_kernel_conv_CNN + wx;  //卷积核索引 W_kj
                int addr4 = addr2 + wy*width_image_input_CNN + wx;     //input中的像素索引 input k
                delta_weight_C1[addr3] = delta_neuron_C1[index] * data_single_image[addr4];
            }
        }
    }
}

//g 6
__kernel void kernel_backward_input_bias( __global float* delta_bias_C1, __global float * delta_neuron_C1) 
{
    int outc = get_global_id(0);
	float tmpb = 0;
    for (int y = 0; y < height_image_C1_CNN; y++) {
			for (int x = 0; x < width_image_C1_CNN; x++) {
				int index = (outc*height_image_C1_CNN*width_image_C1_CNN) + y*width_image_C1_CNN + x;
                tmpb += delta_neuron_C1[index] * 6 * 5 * 5;
            }
    }
	delta_bias_C1[outc] = tmpb;
}



__kernel void  kernel_backward_input(
	__global float *in, //delta_neuron_C1
	__global float *neuron_input, //data_single_image(in)
	__global float *weight_C1, //weight_C1(in) 
	// __global float *delta_weight, // delta_weight_C1
	// __global float *delta_bias,	 // delta_bias_C1
	__global float *out, //delta_neuron_input
	int index // index of data_single_image
){
	//[32, 32]
	int yy = get_global_id(0);
	int xx = get_global_id(1);
	__global float *data_single_image = neuron_input + index;
	int addr4 = yy*32+xx;
	float out_addr4 = 0;
	float data_single_image_addr4=data_single_image[addr4];

	for (int outc = 0; outc < 6; outc++) {
		int addr1 = 5*5*outc; //找到对应的卷积核
		for(int y = max(yy-4,0);y<=min(yy,27);y++){
			int wy = yy - y;
			for(int x = max(xx-4,0);x<=min(xx,27);x++){
				int wx = xx - x;
				int index = (outc*28*28) + y*28 + x; 
				int addr3 = addr1 + wy*5 + wx;  //卷积核索引 W_kj
				out_addr4 += in[index] * weight_C1[addr3] * (1-data_single_image_addr4*data_single_image_addr4);
			}
		}
	}
	out[addr4] = out_addr4;
}


// __kernel void  kernel_backward_input_weight(
// 	__global float *in, //delta_neuron_C1
// 	__global float *neuron_input, //data_single_image(in)
// 	// __global float *weight_C1, //weight_C1(in) 
// 	__global float *delta_weight, // delta_weight_C1
// 	// __global float *delta_bias,	 // delta_bias_C1
// 	// __global float *out, //delta_neuron_input
// 	int index // index of data_single_image
// ){
// 	//[6,5,5]
// 	// int outc = get_global_id(0);
// 	// int wx = get_global_id(1);
// 	// int wy = get_global_id(2);
// 	// __global float *data_single_image = neuron_input + index;
// 	// int addr3 = 25*outc + wy*5 + wx;  //卷积核索引 W_kj
// 	// float delta_weight_addr3 = 0;
// 	// for (int y = 0; y < 28; y++) {
// 	// 	for (int x = 0; x < 28; x++) {
// 	// 		int index = (outc*28*28) + y*28 + x;  //C1 当前神经元 j
// 	// 		int addr2 = y * 32 + x;  //input k
// 	// 		int addr4 = addr2 + wy*32 + wx;     //input中的像素索引 input k
// 	// 		delta_weight_addr3 += in[index] * data_single_image[addr4];
// 	// 	}
// 	// }
// 	// // printf("write:%d->%.6f\n", addr3,delta_weight_addr3);
// 	// delta_weight[addr3]=delta_weight_addr3;
// 	//[6,28,28]
// 	//[1,28,28]
// 	int outc = get_global_id(0);
// 	int y = get_global_id(1);
// 	int x = get_global_id(2);
// 	// printf("0:%d %d %d\n", outc, y, x);
// 	__global float *data_single_image = neuron_input + index;
// 	__local float w_tmp[28*28*15];
// 	int in_index = (outc*28*28) + y*28 + x;
// 	int addr2 = y * 32 + x;  //input k

// 	for (int wy = 0; wy < 3; wy++) {
// 		for (int wx = 0; wx < 5; wx++) {
// 			// int addr3 = 25*outc + wy*5 + wx;  //卷积核索引 W_kj
// 			int addr4 = addr2 + wy*32 + wx;     //input中的像素索引 input k
// 			w_tmp[y*28*15+x*15+wy*5 + wx] = in[in_index] * data_single_image[addr4];
// 			// printf("write:%d,%d->%.6f\n", outc,y*28*15+x*15+wy*5 + wx,w_tmp[y*28*15+x*15+wy*5 + wx]);
// 		}
// 	}
// 	barrier(CLK_LOCAL_MEM_FENCE);
// 	if(x == y && x < 15){
// 		private float tmp = 0;
// 		for(int i=0;i<28;i++){
// 			for(int j=0;j<28;j++){
// 				tmp += w_tmp[i*28*15+j*15+x];
// 			}
// 		}
// 		// printf("1:%d %d %d\n", outc, x, y);
// 		// printf("write:%d->%.6f\n", outc*25+x,tmp);
// 		delta_weight[outc*25+x] = tmp;
// 	}
// 	barrier(CLK_LOCAL_MEM_FENCE);

// 	for (int wy = 3; wy < 5; wy++) {
// 		for (int wx = 0; wx < 5; wx++) {
// 			// int addr3 = 25*outc + wy*5 + wx;  //卷积核索引 W_kj
// 			int addr4 = addr2 + wy*32 + wx;     //input中的像素索引 input k
// 			w_tmp[y*28*10+x*10+(wy-3)*5 + wx] = in[in_index] * data_single_image[addr4];
// 		}
// 	}
// 	barrier(CLK_LOCAL_MEM_FENCE);
// 	if(x == y && x < 10){
// 		float tmp = 0;
// 		for(int i=0;i<28;i++){
// 			for(int j=0;j<28;j++){
// 				tmp += w_tmp[i*28*10+j*10+x];
// 			}
// 		}
// 		// printf("2:%d %d %d\n", outc, x, y);
// 		// printf("write:%d->%.6f\n", outc*25+x+15,tmp);
// 		delta_weight[outc*25+x+15] = tmp;
// 	}
// }

// __kernel void  kernel_backward_input_bias(
// 	__global float *in, //delta_neuron_C1
// 	// __global float *neuron_input, //data_single_image(in)
// 	// __global float *weight_C1, //weight_C1(in) 
// 	// __global float *delta_weight, // delta_weight_C1
// 	__global float *delta_bias	 // delta_bias_C1
// 	// __global float *out, //delta_neuron_input
// 	// int index // index of data_single_image
// ){
// 	//[6]
// 	int outc = get_global_id(0);
// 	// __global float *data_single_image = neuron_input + index;
// 	float delta_bias_outc = 0;
// 	for (int y = 0; y < 28; y++) {
// 		for (int x = 0; x < 28; x++) {
// 			int index = (outc*28*28) + y*28 + x;  //C1 当前神经元 j
// 			delta_bias_outc += in[index]*25;
// 		}
// 	}
// 	delta_bias[outc] = delta_bias_outc;
// }

__kernel void kernel_update_weights(
	__global float * delta,
	__global float * e_weight,
	__global float * weight
){
	int i = get_global_id(0);
	float delta_tmp = delta[i];
	float e_weight_tmp = e_weight[i];
	e_weight_tmp += delta_tmp * delta_tmp;
	weight[i] -= 0.01 * delta_tmp / (sqrt(e_weight_tmp) + 1e-8);
	e_weight[i] = e_weight_tmp;
}

__kernel void  kernel_forward_c1(__global float *in,
                      __global float  *weight,
                      __global float  *bias,
                      __global float  *out,
					  int input_index)
{
	// printf("%d\n",input_index);
    //[6,28,28]
    //[1,7,7]
	//-DfilterSize=5 -DBlockSize=7

	int channel = get_global_id(0);
	int out_height = 28;
	int out_width = 28;
    int  y = get_global_id(1);
    int  x = get_global_id(2);
	int kernel_width = 5;
	// printf("0:%d %d %d\n", channel, y, x);
	int kernel_height = 5;
	int in_width = 32;
	int in_height = 32;
	int in_num = 1;
    int index = (channel*out_height*out_width) + y*out_width + x;
	float sum = 0.0;
	int inc = 0;
	int wx = 0;
	int wy = 0;
	float out_val = 0.0;
	out[index] = 0.0;
	for (inc=0; inc<in_num; inc++) {
        int addr1 = (in_num * channel + inc) * kernel_height * kernel_width;
		int addr2 = (inc)*in_width*in_height;
		__global const float* pw = weight + addr1;   //卷积核
		__global const float* pi = in + input_index + addr2;       //输入图像
		sum = 0.0;
		__global const float* ppw = pw;
		__global const float* ppi = pi + y * in_width + x;
        for(wy = 0; wy < kernel_height; wy++)  {
			for(wx = 0; wx < kernel_width; wx++) {
                sum += *ppw++ * ppi[wy * in_width + wx];
		    }
	     }
	     out[index] += sum;
	}
	out[index] += bias[channel];
	out[index] = tanh((float)(out[index]));
	//out[index] = out_val;
}


// __kernel void  kernel_forward_s2(__global float *in,
//                       __global float  *weight,
//                       __global float  *bias,
//                       __global float  *out)
// {
// 	int channel = get_global_id(0);
// 	int out_height = 14;
// 	int out_width = 14;
// 	int kernel_width=2;
// 	int kernel_height=2;
// 	int in_width=28;
// 	int in_height=28;
// 	//TODO
//     int  y = get_global_id(1);
//     int  x = get_global_id(2);
//     //float scale_factor = 1.0 / (kernel_width * kernel_height);
//     int block = in_width * in_height * channel;
//     int rows = y * kernel_width;
// 	int cols = x * kernel_height;
// 	int index = (channel*out_height*out_width) + y*out_width + x;
// 	out[index] = 0.0;
// 	float out_index=0.0;
// 	for (int m = 0; m < kernel_width; m++) {
// 		for (int n = 0; n < kernel_height; n++) {
//             out[index] += weight[channel] * in[(rows + m) * in_width + cols + n + block];
// 		}
// 	}
// 	out[index] *= 0.25;  //scale_factor;
// 	out[index] += bias[channel] ;
// 	out[index] = tanh((float)(out[index]));
// }

// __kernel void  kernel_forward_c3(__global float *in,
//                       __global float  *weight,
//                       __global float  *bias,
//                       __global float  *out)
// {
// 	//[16,10,10]
// 	//[1,10,10]
// 	int channel = get_global_id(0);
// 	int out_height = 10;
// 	int out_width = 10;
//     int y = get_global_id(1);
//     int x = get_global_id(2);
// 	int kernel_width = 5;
// 	int kernel_height = 5;
// 	int in_width = 14;
// 	int in_height = 14;
// 	int in_num = 6;
//     int index = (channel*out_height*out_width) + y*out_width + x;
// 	float sum = 0.0;
// 	int inc = 0;
// 	int wx = 0;
// 	int wy = 0;
// 	float out_index = 0.0;
// 	out[index] = 0.0;
// 	for (inc=0; inc<in_num; inc++) {
// 		if (!tbl[inc][channel]) continue;
//         int addr1 = (in_num * channel + inc) * kernel_height * kernel_width;
// 		int addr2 = (inc)*in_width*in_height;
// 		__global const float* pw = weight + addr1;   //卷积核
// 		__global const float* pi = in + addr2;       //输入图像
// 		sum = 0.0;
// 		__global const float* ppw = pw;
// 		__global const float* ppi = pi + y * in_width + x;
//         for(wy = 0; wy < kernel_height; wy++)  {
// 			for(wx = 0; wx < kernel_width; wx++) {
//                 sum += *ppw++ * ppi[wy * in_width + wx];
// 		    }
// 	     }
// 	     out[index] += sum;
// 	}
// 	out[index] += bias[channel];
// 	out[index] = tanh((float)(out[index]));
// 	//out[index] = out_index;
// }

// __kernel void  kernel_forward_s4(__global float *in,
//                       __global float  *weight,
//                       __global float  *bias,
//                       __global float  *out)
// {
// 	int channel = get_global_id(0);
// 	int out_height = 5;
// 	int out_width = 5;
//     int  y = get_global_id(1);
//     int  x = get_global_id(2);
// 	int kernel_width=2;
// 	int kernel_height=2;
// 	int in_width=10;
// 	int in_height=10;
//     //float scale_factor = 1.0 / (kernel_width * kernel_height);
//     int block = in_width * in_height * channel;
//     int rows = y * kernel_width;
// 	int cols = x * kernel_height;
// 	int index = (channel*out_height*out_width) + y*out_width + x;
// 	out[index] = 0.0;
// 	float out_index = 0.0;
// 	for (int m = 0; m < kernel_width; m++) {
// 		for (int n = 0; n < kernel_height; n++) {
//             out[index] += weight[channel] * in[(rows + m) * in_width + cols + n + block];
// 		}
// 	}
// 	out[index] *= 0.25;  //scale_factor;
// 	out[index] += bias[channel] ;
// 	out[index] = tanh((float)(out[index]));
// }

// __kernel void  kernel_forward_c5(__global float *in,
//                       __global float  *weight,
//                       __global float  *bias,
//                       __global float  *out)
// {
// 	int channel = get_global_id(0);
//     // int  y = get_global_id(1);
//     // int  x = get_global_id(2);
// 	int out_height=1;
// 	int out_width=1;
// 	int kernel_width = 5;
// 	int kernel_height = 5;
// 	int in_width = 5;
// 	int in_height = 5;
// 	int in_num=16;

// 	int  index = channel*out_height*out_width;
// 	// int  index = (channel*out_height*out_width) + y*out_width + x;
// 	float sum = 0.0;
// 	int inc = 0;
// 	int wx = 0;
// 	int wy = 0;
// 	out[index] = 0.0;
// 	float out_index=0;
// 	for (inc=0; inc<in_num; inc++) {
//         int addr1 = (in_num * channel + inc) * kernel_height * kernel_width;
// 		int addr2 = (inc)*in_width*in_height;
// 		__global const float* pw = weight + addr1;   //卷积核
// 		__global const float* pi = in + addr2;       //输入图像
// 		sum = 0.0;
// 		__global const float* ppw = pw;
// 		__global const float* ppi = pi;
//         for(wy = 0; wy < kernel_height; wy++)  {
// 			for(wx = 0; wx < kernel_width; wx++) {
//                 sum += *ppw++ * ppi[wy * in_width + wx];
// 		    }
// 	     }
// 	     out[index] += sum;
// 	}
// 	out[index] += bias[channel];
// 	out[index] = tanh((float)(out[index]));
// }

__kernel void  kernel_forward_output(__global float *in,
                      __global float  *weight,
                      __global float  *bias,
                      __global float  *out)
{
	int channel = get_global_id(0);
	int in_num=120;
	// out[channel] = 0.0;
	float out_channel = 0.0;
	for (int c = 0; c < in_num; c++) {
		out_channel += weight[c * num_neuron_output_CNN + channel] * in[c];
	}
	out_channel += bias[channel];
	out[channel] = tanh((float)(out_channel));

}

// __kernel void  kernel_backward_output(
// 	__global float *in, //neuron_output
// 	__global float *label, //data_single_label
// 	__global float *out, //delta_neuron_output
// 	int index //index of label
// )
// {
// 	//[10]
// 	int i = get_global_id(0);
// 	__global float *labels = label + index;

// 	float res = (in[i] - labels[i]) * (1.0 - in[i] * in[i]);

// 	out[i] = res;
// }

// __kernel void  kernel_backward_c5(
// 	__global float *in, //delta_neuron_output
// 	__global float *neuron_C5, //neuron_C5(in)
// 	__global float *weight_output, //weight_output(in) 
// 	__global float *delta_weight, // delta_weight_output
// 	__global float *delta_bias,	 // delta_bias_output
// 	__global float *out //delta_neuron_C5
// )
// {
// 	//[120]
// 	int channel = get_global_id(0);
// 	float out_channel = 0.0;
// 	for (int j = 0; j < num_neuron_output_CNN; j++) {
// 		int addr1 = channel * num_neuron_output_CNN + j;    //当前权重
// 		out_channel += in[j] * weight_output[addr1] * (1.0-neuron_C5[channel]*neuron_C5[channel]);
// 		delta_weight[addr1] = in[j] * neuron_C5[channel];
// 		// delta_bias[j] += in[j];
// 	}
// 	out[channel] = out_channel;
// 	if(channel < 10){
// 		delta_bias[channel] = 120*in[channel];
// 	}
// }

// __kernel void  kernel_backward_s4(
// 	__global float *in, //delta_neuron_C5
// 	__global float *neuron_S4, //neuron_S4(in)
// 	__global float *weight_C5, //weight_C5(in) 
// 	__global float *delta_weight, // delta_weight_C5
// 	__global float *delta_bias,	 // delta_bias_C5
// 	__global float *out //delta_neuron_S4
// ){
// 	//[16,5,5]
// 	//[1,5,5]
// 	int inc = get_global_id(0);
// 	int wy = get_global_id(1);
// 	int wx = get_global_id(2);
// 	int addr2 = height_image_S4_CNN*width_image_S4_CNN*inc;   //找到对应的S4输入
// 	int addr4 = addr2 + wy*width_image_S4_CNN + wx;     //S4中的像素索引 S4 k
// 	float out_addr4=0;
// 	float neuron_S4_addr4 = neuron_S4[addr4];
// 	for (int outc = 0; outc < num_map_C5_CNN; outc++) {
// 		int addr1 = width_kernel_conv_CNN*height_kernel_conv_CNN*(num_map_S4_CNN * outc + inc); //找到对应的卷积核

// 		int addr3 = addr1 + wy*width_kernel_conv_CNN + wx;  //卷积核索引 W_kj
// 		out_addr4 += in[outc] * weight_C5[addr3] * (1.0 - neuron_S4_addr4 * neuron_S4_addr4);
// 		delta_weight[addr3] = in[outc] * neuron_S4_addr4;
// 		// delta_bias[outc] += in[outc];
// 		if(inc == 0 && wx == 0 && wy == 0)
// 			delta_bias[outc] = in[outc]*400;
// 	}
// 	out[addr4] = out_addr4;
// }

// __kernel void  kernel_backward_c3(
// 	__global float *in, //delta_neuron_S4
// 	__global float *neuron_C3, //neuron_C3(in)
// 	__global float *weight_S4, //weight_S4(in) 
// 	__global float *delta_weight, // delta_weight_S4
// 	__global float *delta_bias,	 // delta_bias_S4
// 	__global float *out //delta_neuron_C3
// ){
// 	// [16,5,5]
// 	// [1,5,5]
// 	int outc = get_global_id(0);
// 	int y = get_global_id(1);
// 	int x = get_global_id(2);
// 	const float scale_factor = 0.25f;
// 	int block = 10 * 10 * outc; //C3
// 	int index = (outc*5*5) + y*5 + x; //S4 当前神经元j
// 	__local float w_tmp[5][5];
// 	__local float b_tmp[5][5];
// 	w_tmp[y][x] = 0;
// 	// delta_weight[outc] = 0.0f;
// 	// delta_bias[outc] = 0.0f;
// 	// for(int i=0;i<10 * 10;i++){
// 	// 	out[outc * 100 + i] = 0.0f;
// 	// }
// 	for (int m = 0; m < 2; m++) {
// 		for (int n = 0; n < 2; n++) {
// 			int addr2 = block + (y * 2 + m) * 10 + x * 2 + n; //C3 神经元 k
// 			out[addr2] = in[index] * weight_S4[outc] * (1.0 - neuron_C3[addr2] * neuron_C3[addr2]) * scale_factor;
// 			w_tmp[y][x] += in[index] * neuron_C3[addr2] * scale_factor;
// 		}
// 	}
// 	b_tmp[y][x] = in[index]*4;
// 	barrier(CLK_LOCAL_MEM_FENCE);
// 	if(x == 0 && y == 0){
// 		float tmpb=0,tmpw=0;
// 		for(int yy = 0;yy < 5;yy++)
// 			for(int xx = 0;xx < 5;xx++){
// 				tmpb += b_tmp[yy][xx];
// 				tmpw += w_tmp[yy][xx];
// 			}
// 		delta_weight[outc] = tmpw;
// 		delta_bias[outc] = tmpb;
// 	}
// }

// __kernel void  kernel_backward_s2_weight(
// 	__global float *in, //delta_neuron_C3
// 	__global float *neuron_S2, //neuron_S2(in)
// 	// __global float *weight_C3 //weight_C3(in) 
// 	__global float *delta_weight // delta_weight_C3
// 	// __global float *delta_bias,	 // delta_bias_C3
// 	// __global float *out //delta_neuron_S2
// ){
// 	//[25,16,6]
// 	int wxy = get_global_id(0);
// 	int outc = get_global_id(1);
// 	int inc = get_global_id(2);
// 	int wy = wxy / 5;
// 	int wx = wxy - wy * 5;
// 	if (!tbl[inc][outc]) 
// 		return;
// 	int addr3 = 5*5*(6 * outc + inc) + wxy;
// 	float delta_weight_addr3 = 0;

// 	for (int y = 0; y < 10; y++) {
// 		for (int x = 0; x < 10; x++) {
// 			int index = (outc*10*10) + y*10 + x;  //C3 当前神经元 j
// 			int addr2 = 14*14*inc +  y * 14 + x + wy*14 + wx;   //找到对应的S2输入
// 			delta_weight_addr3 += in[index] * neuron_S2[addr2];
// 		}
// 	}
// 	delta_weight[addr3] = delta_weight_addr3;
// }

// __kernel void  kernel_backward_s2_bias(
// 	__global float *in, //delta_neuron_C3
// 	// __global float *neuron_S2, //neuron_S2(in)
// 	// __global float *weight_C3 //weight_C3(in) 
// 	// __global float *delta_weight // delta_weight_C3
// 	__global float *delta_bias	 // delta_bias_C3
// 	// __global float *out //delta_neuron_S2
// ){
// 	//[16]
// 	int outc = get_global_id(0);
// 	float delta_bias_outc = 0;

// 	for (int inc = 0; inc < 6; inc++) {
// 		if (!tbl[inc][outc]) continue;
// 		for (int y = 0; y < 10; y++) {
// 			for (int x = 0; x < 10; x++) {
// 				int index = (outc*10*10) + y*10 + x;  //C3 当前神经元 j
// 				delta_bias_outc += in[index]*25;
// 			}
// 		}
// 	}
// 	delta_bias[outc] = delta_bias_outc;
// }

// __kernel void  kernel_backward_c1(
// 	__global float *in, //delta_neuron_S2
// 	__global float *neuron_C1, //neuron_C1(in)
// 	__global float *weight_S2, //weight_S2(in) 
// 	__global float *delta_weight, // delta_weight_S2
// 	__global float *delta_bias,	 // delta_bias_S2
// 	__global float *out //delta_neuron_C1
// ){
// 	//[6,14,14]
// 	//[1,14,14]
// 	int outc = get_global_id(0);
// 	int y = get_global_id(1); 
// 	int x = get_global_id(2); 
// 	const float scale_factor = 0.25f;
// 	int block = 28*28*outc;
// 	int index = (outc*14*14) + y*14 + x;
// 	__local float w_tmp[14][14];
// 	__local float b_tmp[14][14];
// 	w_tmp[y][x] = 0;

// 	for (int m = 0; m < 2; m++) {
// 		for (int n = 0; n < 2; n++) {
// 			int addr2 = block + (y * 2 + m) * 28 + x * 2 + n;
// 			out[addr2] = in[index] * weight_S2[outc]
// 			* (1-neuron_C1[addr2]*neuron_C1[addr2]) * scale_factor;
// 			w_tmp[y][x]+=in[index] * neuron_C1[addr2] * scale_factor;
// 		}
// 	}
// 	b_tmp[y][x] = in[index] * 4;
// 	barrier(CLK_LOCAL_MEM_FENCE);
// 	if(x == 0 && y == 0){
// 		float tmpb=0,tmpw=0;
// 		for(int yy = 0;yy < 14;yy++)
// 			for(int xx = 0;xx < 14;xx++){
// 				tmpb += b_tmp[yy][xx];
// 				tmpw += w_tmp[yy][xx];
// 			}
// 		delta_weight[outc] = tmpw;
// 		delta_bias[outc] = tmpb;
// 	}
// 	// delta_weight[outc] += in[index] * neuron_C1[addr2] * scale_factor;
// 	// delta_bias[outc] += in[index];
// }






// //g 16
// __kernel void kernel_backward_C3_Wb(__global float* delta_weight_S4, __global float* delta_bias_S4, 
//                                     __global float * delta_neuron_S4,
//                                     __global float * neuron_C3)
// {
//     int outc = get_global_id(0);
//     float scale_factor = 1.0 / (width_kernel_pooling_CNN * height_kernel_pooling_CNN);
//     int block = width_image_C3_CNN * height_image_C3_CNN * outc; //C3
//     for (int y=0; y<height_image_S4_CNN; y++) {
//         for (int x=0; x<width_image_S4_CNN; x++) {
//             int rows = y * width_kernel_pooling_CNN;
//             int cols = x * height_kernel_pooling_CNN;
//             int index = (outc*height_image_S4_CNN*width_image_S4_CNN) + y*width_image_S4_CNN + x; //S4 当前神经元j

//             for (int m = 0; m < height_kernel_pooling_CNN; m++) {
//                 for (int n = 0; n < width_kernel_pooling_CNN; n++) {
//                     int addr1 = outc;  // 权重
//                     int addr2 = block + (rows + m) * width_image_C3_CNN + cols + n; //C3 神经元 k
//                     int addr3 = outc;
//                     delta_weight_S4[addr1] += delta_neuron_S4[index] * neuron_C3[addr2] * scale_factor;
//                     delta_bias_S4[addr3] += delta_neuron_S4[index];
//                 }
//             }
//         }//index
//     }
//     delta_bias_S4[outc] = delta_neuron_S4[outc] * 5 * 5 * 2 * 2;
// }





//----------------------------------------------------
//          constant优化
//-----------------------------------------------------
// __kernel void  kernel_forward_c1(__global float *in,
//                       __constant float  *weight,
//                       __constant float  *bias,
//                       __global float  *out,
// 					  int input_index)
// {
// 	// printf("%d\n",input_index);
//     //[6,28,28]
//     //[1,7,7]
// 	//-DfilterSize=5 -DBlockSize=7

// 	int channel = get_global_id(0);
// 	int out_height = 28;
// 	int out_width = 28;
//     int  y = get_global_id(1);
//     int  x = get_global_id(2);
// 	int kernel_width = 5;
// 	// printf("0:%d %d %d\n", channel, y, x);
// 	int kernel_height = 5;
// 	int in_width = 32;
// 	int in_height = 32;
// 	int in_num = 1;
//     int index = (channel*out_height*out_width) + y*out_width + x;
// 	float sum = 0.0;
// 	int inc = 0;
// 	int wx = 0;
// 	int wy = 0;
// 	float out_val = 0.0;
// 	out[index] = 0.0;
// 	for (inc=0; inc<in_num; inc++) {
//         int addr1 = (in_num * channel + inc) * kernel_height * kernel_width;
// 		int addr2 = (inc)*in_width*in_height;
// 		__constant const float* pw = weight + addr1;   //卷积核
// 		__global const float* pi = in + input_index + addr2;       //输入图像
// 		sum = 0.0;
// 		__constant const float* ppw = pw;
// 		__global const float* ppi = pi + y * in_width + x;
//         for(wy = 0; wy < kernel_height; wy++)  {
// 			for(wx = 0; wx < kernel_width; wx++) {
//                 sum += *ppw++ * ppi[wy * in_width + wx];
// 		    }
// 	     }
// 	     out[index] += sum;
// 	}
// 	out[index] += bias[channel];
// 	out[index] = tanh((float)(out[index]));
// 	//out[index] = out_val;
// }
// 
// __kernel void  kernel_forward_s2(__global float *in,
//                       constant float  *weight,
//                       constant float  *bias,
//                       __global float  *out)
// {
// 	int channel = get_global_id(0);
// 	int out_height = 14;
// 	int out_width = 14;
// 	int kernel_width=2;
// 	int kernel_height=2;
// 	int in_width=28;
// 	int in_height=28;
// 	//TODO
//     int  y = get_global_id(1);
//     int  x = get_global_id(2);
//     //float scale_factor = 1.0 / (kernel_width * kernel_height);
//     int block = in_width * in_height * channel;
//     int rows = y * kernel_width;
// 	int cols = x * kernel_height;
// 	int index = (channel*out_height*out_width) + y*out_width + x;
// 	out[index] = 0.0;
// 	float out_index=0.0;
// 	for (int m = 0; m < kernel_width; m++) {
// 		for (int n = 0; n < kernel_height; n++) {
//             out[index] += weight[channel] * in[(rows + m) * in_width + cols + n + block];
// 		}
// 	}
// 	out[index] *= 0.25;  //scale_factor;
// 	out[index] += bias[channel] ;
// 	out[index] = tanh((float)(out[index]));
// }


// __kernel void  kernel_forward_c5(__global float *in,
//                       __global float  *weight,
//                       constant float  *bias,
//                       __global float  *out)
// {
// 	int channel = get_global_id(0);
//     // int  y = get_global_id(1);
//     // int  x = get_global_id(2);
// 	int out_height=1;
// 	int out_width=1;
// 	int kernel_width = 5;
// 	int kernel_height = 5;
// 	int in_width = 5;
// 	int in_height = 5;
// 	int in_num=16;

// 	int  index = channel*out_height*out_width;
// 	// int  index = (channel*out_height*out_width) + y*out_width + x;
// 	float sum = 0.0;
// 	int inc = 0;
// 	int wx = 0;
// 	int wy = 0;
// 	out[index] = 0.0;
// 	float out_index=0;
// 	for (inc=0; inc<in_num; inc++) {
//         int addr1 = (in_num * channel + inc) * kernel_height * kernel_width;
// 		int addr2 = (inc)*in_width*in_height;
// 		__global const float* pw = weight + addr1;   //卷积核
// 		__global const float* pi = in + addr2;       //输入图像
// 		sum = 0.0;
// 		__global const float* ppw = pw;
// 		__global const float* ppi = pi;
//         for(wy = 0; wy < kernel_height; wy++)  {
// 			for(wx = 0; wx < kernel_width; wx++) {
//                 sum += *ppw++ * ppi[wy * in_width + wx];
// 		    }
// 	     }
// 	     out[index] += sum;
// 	}
// 	out[index] += bias[channel];
// 	out[index] = tanh((float)(out[index]));
// }







// __kernel void  kernel_forward_c5(__global float *in,
//                       __global float  *weight,//constant memory is 64KB
//                       constant float  *bias,
//                       __global float  *out
// 					  )
// {
// 	int channel = get_global_id(0);
//     int  y = get_global_id(1);
//     int  x = get_global_id(2);
// 	int out_height=1;
// 	int out_width=1;
// 	int kernel_width = 5;
// 	int kernel_height = 5;
// 	int in_width = 5;
// 	int in_height = 5;
// 	int in_num=16;
// 	const int num_map_S4_CNN = 16;
// 	//local mem
// 	int tidy=get_local_id(1);
//     int tidx=get_local_id(2);
// 	float local pixel[num_map_S4_CNN][BS5+filtersize-1][BS5+filtersize-1];
// 	for (int i=0; i<in_num; i++){//16
//         int addr2 = i*in_width*in_height;
//         for(int j = 0;j < 5;++ j){
// 			for(int k = 0;k < 5;++ k){
// 				pixel[i][tidy+j][tidx+k] = in[addr2 + (y+j)*in_width + x + k];
// 			}
// 		}
// 	}
//     barrier(CLK_LOCAL_MEM_FENCE);
// 	//
//     int  index = (channel*out_height*out_width) + y*out_width + x;
// 	float sum = 0.0;
// 	int inc = 0;
// 	int wx = 0;
// 	int wy = 0;
// 	out[index] = 0.0;
// 	for (inc=0; inc<in_num; inc++) {
//         int addr1 = (in_num * channel + inc) * kernel_height * kernel_width;
// 		int addr2 = (inc)*(BS5+filtersize-1)*(BS5+filtersize-1);
// 		__global const float* pw = weight + addr1;   //卷积核
// 		__local const float* pi = &pixel[0][0][0] + addr2;       //输入图像
// 		sum = 0.0;
// 		__global const float* ppw = pw;
// 		__local const float* ppi = pi + tidy * (BS5+filtersize-1) + tidx;
//         for(wy = 0; wy < kernel_height; wy++)  {
// 			for(wx = 0; wx < kernel_width; wx++) {
//                 sum += *ppw++ * ppi[wy * (BS5+filtersize-1) + wx];
// 		    }
// 	     }
// 	     out[index] += sum;
// 	}
// 	out[index] += bias[channel];
// 	out[index] = tanh((float)(out[index]));
// }
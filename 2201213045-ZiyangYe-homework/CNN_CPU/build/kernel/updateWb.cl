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
#define eps_CNN						1e-8

#define rate2 (float2)(learning_rate_CNN, learning_rate_CNN)
#define rate8 (float8)(learning_rate_CNN, learning_rate_CNN, learning_rate_CNN, learning_rate_CNN, learning_rate_CNN, learning_rate_CNN, learning_rate_CNN, learning_rate_CNN)
#define rate16 (float16)(learning_rate_CNN, learning_rate_CNN, learning_rate_CNN, learning_rate_CNN, learning_rate_CNN, learning_rate_CNN, learning_rate_CNN, learning_rate_CNN, learning_rate_CNN, learning_rate_CNN, learning_rate_CNN, learning_rate_CNN, learning_rate_CNN, learning_rate_CNN, learning_rate_CNN,learning_rate_CNN)

#define eps2 (float2)(eps_CNN, eps_CNN)
#define eps8 (float8)(eps_CNN, eps_CNN, eps_CNN, eps_CNN, eps_CNN, eps_CNN, eps_CNN, eps_CNN)
#define eps16 (float16)(eps_CNN, eps_CNN, eps_CNN, eps_CNN, eps_CNN, eps_CNN, eps_CNN, eps_CNN, eps_CNN, eps_CNN, eps_CNN, eps_CNN, eps_CNN, eps_CNN, eps_CNN, eps_CNN)

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

#define local_size_c5 (num_neuron_C5_CNN << 1)
#define block_edge3 (num_neuron_output_CNN >> 1)
#define kernel_size (width_kernel_conv_CNN * height_kernel_conv_CNN)
#define scale_factor (1.0/(width_kernel_pooling_CNN * height_kernel_pooling_CNN))
#define gap_c3 (width_image_C3_CNN >> 1)
#define block_edge (width_image_S2_CNN >> 1)
#define gap_c1 (width_image_C1_CNN >> 2)
#define block_edge2 (width_image_input_CNN >> 1)

#define update(delta, e_weight, weight, rate, eps){\
	e_weight += delta * delta;\
	weight -= rate * delta / (sqrt(e_weight) + eps);\
}

//g 65536
__kernel void updateWb(
    __global float * delta_weight_C1,
    __global float * delta_bias_C1,
    __global float * delta_weight_S2,
    __global float * delta_bias_S2,
    __global float * delta_weight_C3,
    __global float * delta_bias_C3,
    __global float * delta_weight_S4,
    __global float * delta_bias_S4,
    __global float * delta_weight_C5,
    __global float * delta_bias_C5,
    __global float * delta_weight_output,
    __global float * delta_bias_output,

    __global float * E_weight_C1,
    __global float * E_bias_C1,
    __global float * E_weight_S2,
    __global float * E_bias_S2,
    __global float * E_weight_C3,
    __global float * E_bias_C3,
    __global float * E_weight_S4,
    __global float * E_bias_S4,
    __global float * E_weight_C5,
    __global float * E_bias_C5,
    __global float * E_weight_output,
    __global float * E_bias_output,

    __global float * weight_C1,
    __global float * bias_C1,
    __global float * weight_S2,
    __global float * bias_S2,
    __global float * weight_C3,
    __global float * bias_C3,
    __global float * weight_S4,
    __global float * bias_S4,
    __global float * weight_C5,
    __global float * bias_C5,
    __global float * weight_output,
    __global float * bias_output
) 
{
    int ii = get_global_id(0);
    if(ii < 150) {
        int i = ii;
        update(delta_weight_C1[i], E_weight_C1[i], weight_C1[i], learning_rate_CNN, eps_CNN);
    }
    else if(ii < 156) {
        int i = ii - 150;
        update(delta_bias_C1[i], E_bias_C1[i], bias_C1[i], learning_rate_CNN, eps_CNN);
    }
    else if(ii < 162) {
        int i = ii - 156;
        update(delta_weight_S2[i], E_weight_S2[i], weight_S2[i], learning_rate_CNN, eps_CNN);
    }
    else if(ii < 168) {
        int i = ii - 162;
        update(delta_bias_S2[i], E_bias_S2[i], bias_S2[i], learning_rate_CNN, eps_CNN);
    }
    else if(ii < 2568) {
        int i = ii - 168;
        update(delta_weight_C3[i], E_weight_C3[i], weight_C3[i], learning_rate_CNN, eps_CNN);
    }
    else if(ii < 2584) {
        int i = ii - 2568;
        update(delta_bias_C3[i], E_bias_C3[i], bias_C3[i], learning_rate_CNN, eps_CNN);
    }
    else if(ii < 2600) {
        int i = ii - 2584;
        update(delta_weight_S4[i], E_weight_S4[i], weight_S4[i], learning_rate_CNN, eps_CNN);
    }
    else if(ii < 2616) {
        int i = ii - 2600;
        update(delta_bias_S4[i], E_bias_S4[i], bias_S4[i], learning_rate_CNN, eps_CNN);
    }
    else if(ii < 50616) {
        int i = ii - 2616;
        update(delta_weight_C5[i], E_weight_C5[i], weight_C5[i], learning_rate_CNN, eps_CNN);
    }
    else if(ii < 50736) {
        int i = ii - 50616;
        update(delta_bias_C5[i], E_bias_C5[i], bias_C5[i], learning_rate_CNN, eps_CNN);
    }
    else if(ii < 51936) {
        int i = ii - 50736;
        update(delta_weight_output[i], E_weight_output[i], weight_output[i], learning_rate_CNN, eps_CNN);
    }
    else if(ii < 51946) {
        int i = ii - 51936;
        update(delta_bias_output[i], E_bias_output[i], bias_output[i], learning_rate_CNN, eps_CNN);
    }
}
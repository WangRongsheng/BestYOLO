#include <chrono>
#include <iostream>
#include <fstream> 
#include <math.h>
#include "../include/yolov5.h"
#include "../include/common.h"

static const int DEVICE  = 0;

Yolov5_detector::Yolov5_detector(const std::string& _engine_file):
                    engine_file(_engine_file)
{
    std::cout<<"engine_file: "<<engine_file<<std::endl;
    init_context();
    std::cout<<"Inference det ["<<input_h<<" x "<<input_w<<"] constructed"<<std::endl;
}

void Yolov5_detector::init_context()
{
    cudaSetDevice(DEVICE);
    char *trtModelStream{nullptr};
    size_t size{0};
    std::ifstream file(engine_file, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    TRTLogger logger;
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr); 
    std::cout<<"Read trt engine success"<<std::endl;
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    std::cout << "deserialize done" << std::endl;

    input_index = engine->getBindingIndex(input_blob_name.c_str());
    auto input_dims = engine->getBindingDimensions(input_index);
    input_c = input_dims.d[1];
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
    input_buffer_size = batchsize * input_c * input_h * input_w * sizeof(float);
    std::cout << "index[" << input_index
            <<"], input shape: "
            <<batchsize
            <<" x "<<input_c
            <<" x "<<input_h
            <<" x "<<input_w
            <<", input_buf_size: "<<input_buffer_size
            <<std::endl;
    CHECK(cudaHostAlloc((void**)&host_input_cpu, input_buffer_size, cudaHostAllocDefault));
    CHECK(cudaMalloc(&device_buffers[input_index], input_buffer_size));

    output_index = engine->getBindingIndex(output_blob_name.c_str());
    auto output_dims = engine->getBindingDimensions(output_index);
    output_numbox = output_dims.d[1];
    output_numprob = output_dims.d[2];
    output_buffer_size = batchsize * output_numbox * output_numprob * sizeof(float);
    std::cout << "index[" << output_index
            <<"], output shape: "
            <<batchsize
            <<" x "<< output_numbox
            <<" x "<< output_numprob
            <<", output_buffer_size: "<<output_buffer_size
            <<std::endl;
    CHECK(cudaHostAlloc((void**)&host_output_cpu, output_buffer_size, cudaHostAllocDefault));
    CHECK(cudaMalloc(&device_buffers[output_index], output_buffer_size));

    CHECK(cudaStreamCreate(&stream));
}

void Yolov5_detector::destroy_context()
{
    bool cudart_ok = true;

    /* Release TensorRT */
    if(context)
    {
        context->destroy();
        context = nullptr;
    }
    if(engine)
    {
        engine->destroy();
        engine = nullptr;
    }
    for(int i = 0; i < 2; ++i)
    {
        if(device_buffers[i])
        {
            CHECK(cudaFree(device_buffers[i]));
        }
    }
    if(host_input_cpu)
        CHECK(cudaFreeHost(host_input_cpu));
    if(host_output_cpu)
        CHECK(cudaFreeHost(host_output_cpu));
}

Yolov5_detector::~Yolov5_detector()
{
    destroy_context();
    std::cout<<"Context destroyed for ["<<input_h<<"x"<<input_w<<"]"<<std::endl;
}

void Yolov5_detector::pre_process(cv::Mat image)
{
    float scale_x = input_w / (float)image.cols;
    float scale_y = input_h / (float)image.rows;
    float scale = std::min(scale_x, scale_y);
    // resize图像，源图像和目标图像几何中心的对齐
    i2d[0] = scale;  i2d[1] = 0;  i2d[2] = (-scale * image.cols + input_w + scale - 1) * 0.5;
    i2d[3] = 0;  i2d[4] = scale;  i2d[5] = (-scale * image.rows + input_h + scale - 1) * 0.5;
    cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);  // image to dst(network), 2x3 matrix
    cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);  // dst to image, 2x3 matrix
    cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);  // 计算一个反仿射变换

    cv::Mat input_image(input_h, input_w, CV_8UC3);
    cv::warpAffine(image, input_image, m2x3_i2d, input_image.size(), \
            cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(114));  // 对图像做平移缩放旋转变换,可逆
    // cv::imwrite("input_image.jpg", input_image);

    std::cout<< "input_image shape: [" << input_image.cols << ", " << input_image.rows << "]"<<std::endl;
    int image_area = input_image.cols * input_image.rows;
    unsigned char* pimage = input_image.data;
    float* phost_b = host_input_cpu + image_area * 0;
    float* phost_g = host_input_cpu + image_area * 1;
    float* phost_r = host_input_cpu + image_area * 2;
    for(int i = 0; i < image_area; ++i, pimage += 3)
    {
        // 注意这里的顺序rgb调换了
        *phost_r++ = pimage[0] / 255.0;
        *phost_g++ = pimage[1] / 255.0;
        *phost_b++ = pimage[2] / 255.0;
    }

    /* upload input tensor and run inference */
    cudaMemcpyAsync(device_buffers[input_index], host_input_cpu, input_buffer_size,
                    cudaMemcpyHostToDevice, stream);
}

void Yolov5_detector::do_detection(cv::Mat& img)
{
    assert(context != nullptr);
    auto start_preprocess = std::chrono::high_resolution_clock::now();
    pre_process(img);
    auto end_preprocess = std::chrono::high_resolution_clock::now();
    float preprocess_time = std::chrono::duration<float, std::milli>(end_preprocess - start_preprocess).count();
    std::cout<<"preprocess time: "<< preprocess_time<<" ms."<< std::endl;
    std::cout<<"Pre-process done!"<<std::endl;

    bool res_ok = true;
    auto t_start1 = std::chrono::high_resolution_clock::now();
    /* Debug device_input on cuda kernel */
    context->enqueue(batchsize, device_buffers, stream, nullptr);
    auto t_end1 = std::chrono::high_resolution_clock::now();
    float total_inf1 = std::chrono::duration<float, std::milli>(t_end1 - t_start1).count();
    std::cout << "Infer take: " << total_inf1/1000 << " s." << std::endl;

    post_process(img);
    std::cout<<"Post-process done!"<<std::endl;
}

void Yolov5_detector::post_process(cv::Mat& img)
{
    CHECK(cudaMemcpyAsync(host_output_cpu, device_buffers[output_index],
                    output_buffer_size, cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));

    std::vector<std::vector<float>> bboxes;

    for (int i = 0; i < output_numbox; i++)
    {
        float* ptr = host_output_cpu + i * output_numprob;
        float objness = ptr[4];
        if(objness < conf_thresh)
            continue;

        float* pclass = ptr + 5;
        int label     = std::max_element(pclass, pclass + num_classes) - pclass;
        float prob    = pclass[label];
        float confidence = prob * objness;
        if(confidence < conf_thresh)
            continue;

        float cx     = ptr[0];
        float cy     = ptr[1];
        float width  = ptr[2];
        float height = ptr[3];
        float left   = cx - width * 0.5;
        float top    = cy - height * 0.5;
        float right  = cx + width * 0.5;
        float bottom = cy + height * 0.5;
        float image_base_left   = d2i[0] * left   + d2i[2];
        float image_base_right  = d2i[0] * right  + d2i[2];
        float image_base_top    = d2i[0] * top    + d2i[5];
        float image_base_bottom = d2i[0] * bottom + d2i[5];
        bboxes.push_back({image_base_left, image_base_top, image_base_right, image_base_bottom, (float)label, confidence});
    }
    printf("decoded bboxes.size = %d\n", bboxes.size());

    // nms
    std::sort(bboxes.begin(), bboxes.end(), [](std::vector<float>& a, std::vector<float>& b){return a[5] > b[5];});
    std::vector<bool> remove_flags(bboxes.size());
    std::vector<std::vector<float>> box_result;
    box_result.reserve(bboxes.size());

    auto iou = [](const std::vector<float>& a, const std::vector<float>& b){
        float cross_left   = std::max(a[0], b[0]);
        float cross_top    = std::max(a[1], b[1]);
        float cross_right  = std::min(a[2], b[2]);
        float cross_bottom = std::min(a[3], b[3]);

        float cross_area = std::max(0.0f, cross_right - cross_left) * std::max(0.0f, cross_bottom - cross_top);
        float union_area = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]) 
                         + std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]) - cross_area;
        if(cross_area == 0 || union_area == 0) return 0.0f;
        return cross_area / union_area;
    };

    for(int i = 0; i < bboxes.size(); ++i){
        if(remove_flags[i]) continue;

        auto& ibox = bboxes[i];
        box_result.emplace_back(ibox);
        for(int j = i + 1; j < bboxes.size(); ++j){
            if(remove_flags[j]) continue;

            auto& jbox = bboxes[j];
            if(ibox[4] == jbox[4]){
                // class matched
                if(iou(ibox, jbox) >= nms_thresh)
                    remove_flags[j] = true;
            }
        }
    }
    printf("box_result.size = %d\n", box_result.size());

    for(int i = 0; i < box_result.size(); ++i){
        auto& ibox = box_result[i];
        float left = ibox[0];
        float top = ibox[1];
        float right = ibox[2];
        float bottom = ibox[3];
        int class_label = ibox[4];
        float confidence = ibox[5];
        cv::Scalar color;
        std::tie(color[0], color[1], color[2]) = random_color(class_label);
        cv::rectangle(img, cv::Point(left, top), cv::Point(right, bottom), color, 3);

        auto name      = cocolabels[class_label];
        auto caption   = cv::format("%s %.2f", name, confidence);
        int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(img, cv::Point(left-3, top-33), cv::Point(left + text_width, top), color, -1);
        cv::putText(img, caption, cv::Point(left, top-5), 0, 1, cv::Scalar::all(0), 2, 16);
    }
    // cv::imwrite("image-draw.jpg", img);
}
/*
目标检测器
1. YoloDet：YOLO目标检测器
2. ArmorDet：装甲板检测器
3. PnPSolver：PnP解算器
*/

#include "detector.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <time.h>
#include <fstream>

using namespace std;

namespace auto_aim
{

    YOLOv8RT::YOLOv8RT(const std::string &engine_file_path)
    {
        std::ifstream file(engine_file_path, std::ios::binary);
        assert(file.good());
        file.seekg(0, std::ios::end);
        auto size = file.tellg();
        file.seekg(0, std::ios::beg);
        char *trtModelStream = new char[size];
        file.read(trtModelStream, size);
        file.close();
        initLibNvInferPlugins(&this->gLogger, "");
        this->runtime = nvinfer1::createInferRuntime(this->gLogger);
        assert(this->runtime != nullptr);

        this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
        assert(this->engine != nullptr);
        delete[] trtModelStream;
        this->context = this->engine->createExecutionContext();
        assert(this->context != nullptr);
        
        cudaStreamCreate(&this->stream);

#ifdef TENSORRT10
        this->num_bindings = this->engine->getNbIOTensors();
#else
        this->num_bindings = this->num_bindings = this->engine->getNbBindings();
#endif

        for (int i = 0; i < this->num_bindings; ++i)
        {
            det::Binding binding;
            nvinfer1::Dims dims;

#ifdef TENSORRT10
            std::string name = this->engine->getIOTensorName(i);
            nvinfer1::DataType dtype = this->engine->getTensorDataType(name.c_str());
#else
            nvinfer1::DataType dtype = this->engine->getBindingDataType(i);
            std::string name = this->engine->getBindingName(i);
#endif
            binding.name = name;
            binding.dsize = type_to_size(dtype);

#ifdef TENSORRT10
            bool IsInput = engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
#else
            bool IsInput = engine->bindingIsInput(i);
#endif
            if (IsInput)
            {
                this->num_inputs += 1;

#ifdef TENSORRT10
                dims = this->engine->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
                // set max opt shape
                this->context->setInputShape(name.c_str(), dims);
#else
                dims = this->engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
                // set max opt shape
                this->context->setBindingDimensions(i, dims);
#endif

                binding.size = get_size_by_dims(dims);
                binding.dims = dims;
                this->input_bindings.push_back(binding);
            }
            else
            {
#ifdef TENSORRT10
                dims = this->context->getTensorShape(name.c_str());
#else
                dims = this->context->getBindingDimensions(i);
#endif

                binding.size = get_size_by_dims(dims);
                binding.dims = dims;
                this->output_bindings.push_back(binding);
                this->num_outputs += 1;
            }
        }
    }

    YOLOv8RT::~YOLOv8RT()
    {
#ifdef TENSORRT10
        delete this->context;
        delete this->engine;
        delete this->runtime;
#else
        this->context->destroy();
        this->engine->destroy();
        this->runtime->destroy();
#endif

        cudaStreamDestroy(this->stream);
        for (auto &ptr : this->device_ptrs)
        {
            CHECK(cudaFree(ptr));
        }

        for (auto &ptr : this->host_ptrs)
        {
            CHECK(cudaFreeHost(ptr));
        }
    }

    void YOLOv8RT::make_pipe()
    {

        for (auto &bindings : this->input_bindings)
        {
            void *d_ptr;
            CHECK(cudaMalloc(&d_ptr, bindings.size * bindings.dsize));                  // CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->stream)); cuda11
            this->device_ptrs.push_back(d_ptr);

#ifdef TENSORRT10
            auto name = bindings.name.c_str();
            this->context->setInputShape(name, bindings.dims);
            this->context->setTensorAddress(name, d_ptr);
#endif
        }

        for (auto &bindings : this->output_bindings)
        {
            void *d_ptr, *h_ptr;
            size_t size = bindings.size * bindings.dsize;
            CHECK(cudaMalloc(&d_ptr, size));                            // CHECK(cudaMallocAsync(&d_ptr, size, this->stream)); cuda11
            CHECK(cudaHostAlloc(&h_ptr, size, 0));
            this->device_ptrs.push_back(d_ptr);
            this->host_ptrs.push_back(h_ptr);
#ifdef TENSORRT10
            auto name = bindings.name.c_str();
            this->context->setTensorAddress(name, d_ptr);
#endif
        }
    }

    void YOLOv8RT::letterbox(const cv::Mat &image, cv::Mat &out, cv::Size &size)
    {
        const float inp_h = size.height;
        const float inp_w = size.width;
        float height = image.rows;
        float width = image.cols;

        float r = std::min(inp_h / height, inp_w / width);
        int padw = std::round(width * r);
        int padh = std::round(height * r);

        cv::Mat tmp;
        if ((int)width != padw || (int)height != padh)
        {
            cv::resize(image, tmp, cv::Size(padw, padh));
        }
        else
        {
            tmp = image.clone();
        }

        float dw = inp_w - padw;
        float dh = inp_h - padh;

        dw /= 2.0f;
        dh /= 2.0f;
        int top = int(std::round(dh - 0.1f));
        int bottom = int(std::round(dh + 0.1f));
        int left = int(std::round(dw - 0.1f));
        int right = int(std::round(dw + 0.1f));

        cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});

        out.create({1, 3, (int)inp_h, (int)inp_w}, CV_32F);

        std::vector<cv::Mat> channels;
        cv::split(tmp, channels);

        cv::Mat c0((int)inp_h, (int)inp_w, CV_32F, (float *)out.data);
        cv::Mat c1((int)inp_h, (int)inp_w, CV_32F, (float *)out.data + (int)inp_h * (int)inp_w);
        cv::Mat c2((int)inp_h, (int)inp_w, CV_32F, (float *)out.data + (int)inp_h * (int)inp_w * 2);

        channels[0].convertTo(c2, CV_32F, 1 / 255.f);
        channels[1].convertTo(c1, CV_32F, 1 / 255.f);
        channels[2].convertTo(c0, CV_32F, 1 / 255.f);

        this->pparam.ratio = 1 / r;
        this->pparam.dw = dw;
        this->pparam.dh = dh;
        this->pparam.height = height;
        this->pparam.width = width;
        ;
    }

    void YOLOv8RT::copy_from_Mat(const cv::Mat &image)
    {
        cv::Mat nchw;
        auto &in_binding = this->input_bindings[0];
        int width = in_binding.dims.d[3];
        int height = in_binding.dims.d[2];
        cv::Size size{width, height};
        this->letterbox(image, nchw, size);

        CHECK(cudaMemcpyAsync(
            this->device_ptrs[0], nchw.ptr(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));
#ifdef TENSORRT10
        auto name = this->input_bindings[0].name.c_str();
        this->context->setInputShape(name, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
        this->context->setTensorAddress(name, this->device_ptrs[0]);
#else
        this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, height, width}});
#endif
    }

    void YOLOv8RT::infer()
    {
#ifdef TENSORRT10
        this->context->enqueueV3(this->stream);
#else
        this->context->enqueueV2(this->device_ptrs.data(), this->stream, nullptr);
#endif
        for (int i = 0; i < this->num_outputs; i++)
        {
            size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
            CHECK(cudaMemcpyAsync(
                this->host_ptrs[i], this->device_ptrs[i + this->num_inputs], osize, cudaMemcpyDeviceToHost, this->stream));
        }
        cudaStreamSynchronize(this->stream);
    }

    void YOLOv8RT::postprocess(std::vector<det::Object> &objs, float score_thres, float iou_thres, int topk, int num_labels)
    {
        objs.clear();
        int num_channels = this->output_bindings[0].dims.d[1];
        int num_anchors = this->output_bindings[0].dims.d[2];

        auto &dw = this->pparam.dw;
        auto &dh = this->pparam.dh;
        auto &width = this->pparam.width;
        auto &height = this->pparam.height;
        auto &ratio = this->pparam.ratio;

        std::vector<cv::Rect> bboxes;
        std::vector<float> scores;
        std::vector<int> labels;
        std::vector<int> indices;

        cv::Mat output = cv::Mat(num_channels, num_anchors, CV_32F, static_cast<float *>(this->host_ptrs[0]));
        output = output.t();
        for (int i = 0; i < num_anchors; i++)
        {
            auto row_ptr = output.row(i).ptr<float>();
            auto bboxes_ptr = row_ptr;
            auto scores_ptr = row_ptr + 4;
            auto max_s_ptr = std::max_element(scores_ptr, scores_ptr + num_labels);
            float score = *max_s_ptr;
            if (score > score_thres)
            {
                float x = *bboxes_ptr++ - dw;
                float y = *bboxes_ptr++ - dh;
                float w = *bboxes_ptr++;
                float h = *bboxes_ptr;

                float x0 = clamp((x - 0.5f * w) * ratio, 0.f, width);
                float y0 = clamp((y - 0.5f * h) * ratio, 0.f, height);
                float x1 = clamp((x + 0.5f * w) * ratio, 0.f, width);
                float y1 = clamp((y + 0.5f * h) * ratio, 0.f, height);

                int label = max_s_ptr - scores_ptr;
                cv::Rect_<float> bbox;
                bbox.x = x0;
                bbox.y = y0;
                bbox.width = x1 - x0;
                bbox.height = y1 - y0;

                bboxes.push_back(bbox);
                labels.push_back(label);
                scores.push_back(score);
            }
        }

        cv::dnn::NMSBoxes(bboxes, scores, score_thres, iou_thres, indices);

        int cnt = 0;
        for (auto &i : indices)
        {
            if (cnt >= topk)
            {
                break;
            }
            det::Object obj;
            obj.rect = bboxes[i];
            obj.prob = scores[i];
            obj.label = labels[i];
            objs.push_back(obj);
            cnt += 1;
        }
    }

    vector<Armor> ArmorDet::detect(const vector<vector<int>> &results, const cv::Mat &image)
    {
        vector<Armor> armors;
        vector<Armor> armors_;

        // 遍历检测结果
        for (vector<int> result : results)
        {
            // 获取检测结果的ROI
            cv::Rect roi = cv::Rect(result[0], result[1], result[2] - result[0], result[3] - result[1]);
            // 筛选掉超出图像范围的ROI
            if (roi.x < 0 || roi.x + roi.width > image.cols || roi.y < 0 || roi.y + roi.height > image.rows || roi.height < 0 || roi.width < 0)
            {
                continue;
            }
            // 获取ROI图像
            cv::Mat roi_image = image(roi);
            // 获取ROI左上角坐标
            cv::Point2f roi_tl = cv::Point2f(result[0], result[1]);
            // 获取ROI中的灯条
            vector<Light> lights = find_lights(roi_image, roi_tl);
            // 匹配灯条
            vector<Armor> armors = match_lights(lights, result);
            // 筛出无效装甲板
            /*if (armor.center.x != 0 && armor.center.y != 0) {
                armors.push_back(armor);
            }*/
            armors_.insert(armors_.end(), armors.begin(), armors.end());
        }
        return armors_;
    }

    vector<Light> ArmorDet::find_lights(const cv::Mat &roi_image, const cv::Point2f &roi_tl)
    {
        vector<Light> lights;
        cv::Mat gray_image, binary_image;
        // 灰度化、高斯模糊、二值化
        cvtColor(roi_image, gray_image, cv::COLOR_BGR2GRAY);
        /*GaussianBlur(gray_image, gray_image, Size(3, 3), 0);*/
        cv::threshold(gray_image, binary_image, 180, 255, cv::THRESH_BINARY);
        // cv::imshow("binary_image", binary_image);
        // cv::waitKey(1);
        vector<vector<cv::Point>> contours;
        vector<cv::Vec4i> hierarchy;
        // 查找轮廓
        findContours(binary_image, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (vector<cv::Point> contour : contours)
        {
            // 筛选掉小轮廓
            if (contour.size() > 3 && contourArea(contour) > 5)
            {
                // 获取最小外接矩形
                cv::RotatedRect rect = cv::minAreaRect(contour);

                // 判断是否为灯条
                if (is_light(Light(rect)))
                {
                    cv::Point2f vertices[4];
                    rect.points(vertices);
                    // 绘制灯条最小外接矩形
                    /*for (int i = 0; i < 4; i++) {
                        line(roi_image, vertices[i], vertices[(i + 1) % 4], Scalar(0, 0, 255), 2);
                    }*/
                    // 获取灯条的四个顶点真实坐标
                    for (int i = 0; i < 4; i++)
                    {
                        vertices[i].x += roi_tl.x;
                        vertices[i].y += roi_tl.y;
                    }
                    cv::RotatedRect rect_ = cv::RotatedRect(vertices[0], vertices[1], vertices[2]);
                    lights.push_back(Light(rect_));
                }
            }
        }
        if (lights.size() > 2)
        {
            sort(lights.begin(), lights.end(), [](const Light &a, const Light &b)
                 { return a.center.x > b.center.x; });
            // 只保留首尾灯条
            lights.erase(lights.begin() + 1, lights.end() - 1);
        }
        return lights;
    }

    // TODO:
    bool ArmorDet::is_light(const Light &light)
    {
        // 筛选掉不符合条件的灯条
        if (light.width / light.length > 0.5 || light.tilt_angle > 45)
        {
            return false;
        }
        return true;
    }

    vector<Armor> ArmorDet::match_lights(const vector<Light> &lights, vector<int> &result)
    {
        vector<Armor> armors;
        vector<string> class_names = {"red", "blue"};
        // 匹配灯条
        for (size_t i = 0; i < lights.size(); i++)
        {
            for (size_t j = i + 1; j < lights.size(); j++)
            {
                if (is_armor(lights[i], lights[j]) != ArmorType::INVALID)
                {
                    Armor armor(lights[i], lights[j]);
                    armor.classfication_result = class_names[result[4]];
                    armor.color = class_names[result[4]];
                    armor.yolo_confidence = result[5];
                    armors.push_back(armor);
                }
            }
        }
        return armors;
    }

    ArmorType ArmorDet::is_armor(const Light &light1, const Light &light2)
    {
        // 判断两根灯条是否为装甲板
        float length_ratio = light1.length < light2.length ? light1.length / light2.length : light2.length / light1.length;
        float avg_length = (light1.length + light2.length) / 2;
        float center_distance = norm(light1.center - light2.center) / avg_length;
        // TODO: 判断装甲板类型

        cv::Point2f diff = light1.center - light2.center;
        float angle = atan2(abs(diff.y), abs(diff.x)) * 180 / CV_PI;
        // cout << length_ratio << " " << center_distance << " " << angle << endl;
        bool is_armor = length_ratio > 0.4 && center_distance > 1.0 && angle < 45;
        ArmorType type;
        if (is_armor)
        {
            type = center_distance > 3.0 ? ArmorType::LARGE : ArmorType::SMALL;
        }
        else
        {
            type = ArmorType::INVALID;
        }

        return type;
    }

    PnPSolver::PnPSolver(const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs)
    {
        // 初始化相机参数
        this->camera_matrix = camera_matrix;
        this->dist_coeffs = dist_coeffs;
        // 初始化物体坐标
        object_points = {cv::Point3f(0, 0, 0), cv::Point3f(0.055 / 2, -0.135 / 2, 0), cv::Point3f(-0.055 / 2, -0.135 / 2, 0),
                         cv::Point3f(-0.055 / 2, 0.135 / 2, 0), cv::Point3f(0.055 / 2, 0.135 / 2, 0)}; // m为单位
    }

    vector<double> PnPSolver::solve(const Armor &armor)
    {
        // 初始化图像坐标
        image_points = {armor.center, armor.left_light.top, armor.left_light.bottom, armor.right_light.bottom, armor.right_light.top};
        // 解PnP
        cv::solvePnP(object_points, image_points, camera_matrix, dist_coeffs, rvec, tvec);
        // 计算距离
        double distance = sqrt(pow(tvec.at<double>(0), 2) + pow(tvec.at<double>(1), 2) + pow(tvec.at<double>(2), 2));
        // cout << "distance: " << distance << endl;
        // 计算角度
        double yaw = atan2(tvec.at<double>(0), tvec.at<double>(2));
        double pitch = atan2(tvec.at<double>(1), tvec.at<double>(2));
        // cout << "yaw: " << yaw << " pitch: " << pitch << endl;
        vector<double> data = {armor.center.x, armor.center.y, yaw, pitch, distance};
        return data;
    }
}
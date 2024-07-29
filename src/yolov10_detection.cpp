#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

// 加载模型
ov::Core core = ov::Core();
std::shared_ptr<ov::Model> model = core.read_model("F:/items/Archive01/Projects/YOLO/weights/yolov10n_e10_openvino_model/yolov10n_e10.xml", "F:/items/Archive01/Projects/YOLO/weights/yolov10n_e10_openvino_model/yolov10n_e10.bin");
ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
ov::InferRequest infer_request = compiled_model.create_infer_request();
float scale;

static cv::Mat letterbox(const cv::Mat& source) {
    // 将图像填充为正方形
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

static ov::Tensor infer(const cv::Mat& image) {
    // 推理
	cv::Mat letterbox_image = letterbox(image);                 // 将图像填充为正方形
	scale = letterbox_image.size[0] / 640.0;                    // 计算缩放比例
	cv::Mat blob = cv::dnn::blobFromImage(letterbox_image, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true);    // 将图像转换为blob
	auto& input_port = compiled_model.input();                  // 获取输入端口
	ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), blob.ptr(0));    // 创建输入张量
	infer_request.set_input_tensor(input_tensor);               // 设置输入张量
	infer_request.infer();									    // 推理
	auto output = infer_request.get_output_tensor(0);           // 获取输出张量
    return output;
}

static std::vector<std::vector<int>> postprocess(const ov::Tensor& output, const float& score_threshold) {
    // 后处理
	float* data = output.data<float>();                         // 获取输出数据
	cv::Mat output_buffer(output.get_shape()[1], output.get_shape()[2], CV_32F, data);	// 创建输出缓冲区
	//cv::transpose(output_buffer, output_buffer);				// 转置
    std::vector<std::vector<int>> results;
    // 遍历输出层
	for (int i = 0; i < output_buffer.rows; i++) {              // 遍历每个边界框
        // 获取类别得分
        float score = output_buffer.at<float>(i, 4);
		float class_id = output_buffer.at<float>(i, 5);
		if (score > score_threshold) {							// 判断是否满足阈值
            // 获取边界框
            float ltx = output_buffer.at<float>(i, 0);
            float lty = output_buffer.at<float>(i, 1);
            float rbx = output_buffer.at<float>(i, 2);
            float rby = output_buffer.at<float>(i, 3);
            // 计算边界框真实坐标
            int left = int(ltx * scale);
			int top = int(lty * scale);
			int right = int(rbx * scale);
			int bottom = int(rby * scale);
			std::vector<int> box = { left, top, right, bottom, int(class_id), int(score * 100) };
            // 将边界框存储
            results.push_back(box);
        }
    }
    return results;
}

int main()
{
	cv::Mat image = cv::imread("F:/items/Archive01/Projects/YOLO/images/robo_2.jpg");
	ov::Tensor output = infer(image);
	std::vector<std::vector<int>> results = postprocess(output, 0.5);
	for (size_t i = 0; i < results.size(); i++) {
		cv::rectangle(image, cv::Point(results[i][0], results[i][1]), cv::Point(results[i][2], results[i][3]), cv::Scalar(0, 255, 0), 2);
		cv::putText(image, std::to_string(results[i][4]) + " " + std::to_string(results[i][5]) + "%", cv::Point(results[i][0], results[i][1]), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
	}
	cv::imshow("image", image);
	cv::waitKey(0);
	return 0;
}
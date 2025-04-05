#include <serial_thread.hpp>

extern std::atomic<bool> detect_color; // true for red, false for blue
SerialThread::SerialThread(const std::string& port, uint baud_rate) : io_ctx_(), serial_port_(io_ctx_), running_(false) {
    while (true) {
        try {
            serial_port_.open(port);
            serial_port_.set_option(boost::asio::serial_port::baud_rate(baud_rate));
            serial_port_.set_option(boost::asio::serial_port::character_size(8));
            serial_port_.set_option(boost::asio::serial_port::parity(boost::asio::serial_port::parity::none));
            serial_port_.set_option(boost::asio::serial_port::stop_bits(boost::asio::serial_port::stop_bits::one));
            break;
        } catch (const std::exception& e) {
            std::cerr << "Failed to open serial port: " + std::string(e.what()) << ". Retrying in 1 second..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
}


SerialThread::~SerialThread() {
    stop();
}

void SerialThread::start() {
    running_ = true;
    thread_ = std::thread([this]() {
        start_async_read();
        io_ctx_.run();
    });
}

void SerialThread::stop() {
    if (running_) {
        running_ = false;
        io_ctx_.stop();
        if (thread_.joinable()) {
            thread_.join();
        }
    }
}

void SerialThread::send_packet(const SendPacket& packet) {
    auto buffer = serialize_packet(packet);
    io_ctx_.post([this, buffer]() {
        boost::asio::async_write(serial_port_, boost::asio::buffer(buffer),
            [this](boost::system::error_code ec, std::size_t /*bytes_transferred*/) {
                if (ec) {
                    std::cerr << "Write error: " << ec.message() << std::endl;
                }
            });
    });
}

void SerialThread::start_async_read() {
    serial_port_.async_read_some(boost::asio::buffer(read_buf_),
        [this](boost::system::error_code ec, std::size_t bytes_transferred) {
            if (!ec) {
                handle_receive(bytes_transferred);
                start_async_read();
            } else {
                std::cerr << "Read error: " << ec.message() << std::endl;
            }
        });
}

void SerialThread::handle_receive(std::size_t length) {
    rx_buffer_.insert(rx_buffer_.end(), read_buf_, read_buf_ + length);
    while (rx_buffer_.size() >= sizeof(ReceivePacket)) {
        ReceivePacket packet;
        if (deserialize_packet(rx_buffer_.data(), rx_buffer_.size(), packet)) {
            detect_color.store(packet.detect_color == 0 ? true : false);
            rx_buffer_.erase(rx_buffer_.begin(), rx_buffer_.begin() + sizeof(ReceivePacket));
        } else {
            rx_buffer_.erase(rx_buffer_.begin());
        }
    }
}

// 序列化函数
std::vector<uint8_t> SerialThread::serialize_packet(const SendPacket& packet) {
    std::vector<uint8_t> buffer(sizeof(SendPacket));
    auto* ptr = reinterpret_cast<const uint8_t*>(&packet);
    buffer.assign(ptr, ptr + sizeof(SendPacket));
    
    // 计算并填充CRC校验
    crc16::Append_CRC16_Check_Sum(buffer.data(), buffer.size());
    
    // 调试输出
    std::cout << "TX Packet: ";
    for (auto b : buffer) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') 
                 << static_cast<int>(b) << " ";
    }
    std::cout << std::dec << "\n";
    
    return buffer;
}

// 反序列化函数
bool SerialThread::deserialize_packet(const uint8_t* data, size_t length, ReceivePacket& packet) {
    if (length < sizeof(ReceivePacket)) return false;
    
    // 校验CRC
    if (!crc16::Verify_CRC16_Check_Sum(data, sizeof(ReceivePacket))) {
        std::cerr << "CRC check failed\n";
        return false;
    }

    // 调试输出
    std::cout << "RX Packet: ";
    for (size_t i = 0; i < sizeof(ReceivePacket); ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') 
                 << static_cast<int>(data[i]) << " ";
    }
    std::cout << std::dec << "\n";

    packet.header = data[0];
    packet.detect_color = data[1] & 0x01;
    packet.reset_tracker = (data[1] >> 1) & 0x01;
    packet.reserved = (data[1] >> 2) & 0x3f;
    packet.crc_checksum = (0xffff & data[sizeof(ReceivePacket) - 1]) | (data[sizeof(ReceivePacket) - 2] << 8);

    memcpy(&packet, data, sizeof(ReceivePacket));
    return true;
}

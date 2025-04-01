#ifndef SERIAL_THREAD_HPP
#define SERIAL_THREAD_HPP

#include <boost/asio.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include <functional>
#include <queue>
#include <packet.hpp>

extern std::atomic<bool> detect_color; // true for red, false for blue

class SerialThread {
public:

    SerialThread(const std::string& port, uint baud_rate)
        : io_ctx_(), serial_port_(io_ctx_), running_(false) {
        try {
            serial_port_.open(port);
            serial_port_.set_option(boost::asio::serial_port::baud_rate(baud_rate));
            serial_port_.set_option(boost::asio::serial_port::character_size(8));
            serial_port_.set_option(boost::asio::serial_port::parity(boost::asio::serial_port::parity::none));
            serial_port_.set_option(boost::asio::serial_port::stop_bits(boost::asio::serial_port::stop_bits::one));
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to open serial port: " + std::string(e.what()));
        }
    }

    ~SerialThread() {
        stop();
    }

    void start() {
        running_ = true;
        thread_ = std::thread([this]() {
            start_async_read();
            io_ctx_.run();
        });
    }

    void stop() {
        if (running_) {
            running_ = false;
            io_ctx_.stop();
            if (thread_.joinable()) {
                thread_.join();
            }
        }
    }

    void send_packet(const SendPacket& packet) {
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

private:
    void start_async_read() {
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

    void handle_receive(std::size_t length) {
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

    boost::asio::io_context io_ctx_;
    boost::asio::serial_port serial_port_;
    std::thread thread_;
    std::atomic<bool> running_;
    uint8_t read_buf_[128];
    std::vector<uint8_t> rx_buffer_;
};

#endif // SERIAL_THREAD_HPP
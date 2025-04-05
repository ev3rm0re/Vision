#ifndef _SERIAL_THREAD_HPP_
#define _SERIAL_THREAD_HPP_

#include <boost/asio.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include <functional>
#include <queue>

#include <packet.hpp>

class SerialThread {
public:
    SerialThread(const std::string& port, uint baud_rate);
    ~SerialThread();
    void start();
    void stop();
    void send_packet(const SendPacket& packet);

private:
    boost::asio::io_context io_ctx_;
    boost::asio::serial_port serial_port_;
    std::thread thread_;
    std::atomic<bool> running_;
    uint8_t read_buf_[128];
    std::vector<uint8_t> rx_buffer_;

    void start_async_read();
    void handle_receive(std::size_t length);
    std::vector<uint8_t> serialize_packet(const SendPacket& packet);
    bool deserialize_packet(const uint8_t* data, size_t length, ReceivePacket& packet);
};

#endif // SERIAL_THREAD_HPP
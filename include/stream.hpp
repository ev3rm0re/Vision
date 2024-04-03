#ifndef STREAM_HPP
#define STREAM_HPP

class Stream
{
    private:
        char* buffer;   // 缓冲区
        int capacity;   // 缓冲区大小
        int start;      // 流的开头
        int length;     // 流的长度
        void expand();  // 扩展缓冲区至原来的两倍

    public:
        Stream(int initCapacity = 16);              // 初始化一个流
        ~Stream();                              // 析构函数
        int getLength();                        // 获取流的长度
        void append(char aChar);                // 向流的末尾追加一个字符
        void append(const char* buf, int len);  // 向流的末尾追加多个字符
        char peek();                            // 查看流的第一个字符
        int peek(char* buf, int len);           // 查看流的前len个字符，返回实际读取的字符数
        char take();                            // 读取流的第一个字符
        int take(char* buf, int len);           // 读取流的前len个字符，返回实际读取的字符数
};
#endif

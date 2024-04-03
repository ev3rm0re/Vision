#ifndef TIMESTAMP_HPP
#define TIMESTAMP_HPP

typedef long long timestamp_t;

class Timestamp {
    public:
        static timestamp_t now();
};

#endif
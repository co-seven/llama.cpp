#include "media-worker.h"

media_worker::media_worker(std::string name) : name_(std::move(name)) {
    thread_ = std::thread(&media_worker::loop, this);
    std::cerr << "[media-worker] backend '" << name_ << "' worker thread started: " << thread_.get_id() << "\n";
}

media_worker::~media_worker() {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        stop_ = true;
    }
    cv_.notify_one();
    if (thread_.joinable()) {
        thread_.join();
    }
    std::cerr << "[media-worker] backend '" << name_ << "' worker thread stopped\n";
}

const std::string & media_worker::name() const {
    return name_;
}

std::thread::id media_worker::thread_id() const {
    return thread_.get_id();
}

void media_worker::loop() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_.wait(lock, [&]() { return stop_ || !queue_.empty(); });
            if (stop_ && queue_.empty()) {
                return;
            }
            task = std::move(queue_.front());
            queue_.pop();
        }
        try {
            task();
        } catch (const std::exception & e) {
            std::cerr << "[media-worker] backend '" << name_ << "' uncaught task failure on thread "
                      << std::this_thread::get_id() << ": " << e.what() << "\n";
        } catch (...) {
            std::cerr << "[media-worker] backend '" << name_ << "' uncaught task failure on thread "
                      << std::this_thread::get_id() << ": unknown exception\n";
        }
    }
}

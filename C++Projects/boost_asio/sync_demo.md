# Sync Client and Service

## Sync Shortage
1. 同步读写的缺陷在于读写是阻塞的，如果客户端对端不发送数据服务器的read操作是阻塞的，这将导致服务器处于阻塞等待状态。
2. 可以通过开辟新的线程为新生成的连接处理读写，但是一个进程开辟的线程是有限的，约为2048个线程，在Linux环境可以通过unlimit增加一个进程开辟的线程数，但是线程过多也会导致切换消耗的时间片较多。
3. 该服务器和客户端为应答式，实际场景为全双工通信模式，发送和接收要独立分开。
4. 该服务器和客户端未考虑粘包处理。

## Client
```cpp
#include <iostream>
#include <boost/asio.hpp>

const int MAX_LENGTH = 1024;

int main() {
	try {
		std::cout << "This is client!\n";
		// client ioc
		boost::asio::io_context ioc;
		boost::asio::ip::tcp::endpoint
			remote_ep(boost::asio::ip::address::from_string("127.0.0.1"), 10086);
		boost::asio::ip::tcp::socket sock(ioc);

		boost::system::error_code ec = boost::asio::error::host_not_found;;

		sock.connect(remote_ep, ec);

		if (ec) {
			std::cout << "connect failed, code is " << ec.value() << " error msg is " << ec.message();
			return 0;
		}

		for (;;) {
			std::cout << "Enter message: ";
			char request[MAX_LENGTH];
			std::cin.getline(request, MAX_LENGTH);

			size_t request_length = strlen(request);
			// sync write; when there is nothing to write, it will block
			boost::asio::write(sock, boost::asio::buffer(request, request_length));

			char reply[MAX_LENGTH];
			// get service reply
			size_t reply_length = boost::asio::read(sock,
				boost::asio::buffer(reply, request_length));
			std::cout << "Reply is: ";
			std::cout.write(reply, reply_length);
			std::cout << "\n";
		}
	}
	catch (std::exception& e) {
		std::cerr << "Exception: " << e.what() << "\n";
	}

	return 0;
}
```

## Service
```cpp
#include <iostream>
#include <boost/asio.hpp>
#include <memory>
#include <set>

const int MAX_LENGTH = 1024;
typedef std::shared_ptr<boost::asio::ip::tcp::socket> socket_ptr;
std::set<std::shared_ptr<std::thread>> thread_set;

void session(socket_ptr sock) {
    // create session for socket to read
    try {
        for (;;) {
            char data[MAX_LENGTH];
            memset(data, '\0', MAX_LENGTH);
            boost::system::error_code error;
            size_t length = sock->read_some(boost::asio::buffer(data, MAX_LENGTH), error);
            if (error == boost::asio::error::eof) {
                std::cout << "connection closed by peer\n";
                break;
            }
            else if (error) {
                throw boost::system::system_error(error);
            }
            std::cout << "receive from " << sock->remote_endpoint().address().to_string() << "\n";
            std::cout << "receive message is " << data << "\n";

            boost::asio::write(*sock, boost::asio::buffer(data, length));
        }
    } catch (std::exception& e) {
        std::cerr << "Exception in thread: " << e.what() << "\n";
    }
}

void server(boost::asio::io_context& ioc, unsigned short port) {
    boost::asio::ip::tcp::acceptor
        server_acceptor(ioc, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port));
    for (;;) {
        // create empty socket for acceptor
        // when the acceptor accept the client socket; this empty socket will be replaced to the client socket
        socket_ptr socket = std::make_shared<boost::asio::ip::tcp::socket>(ioc);
        // sycn accept; if there is no client connection, it will block
        server_acceptor.accept(*socket);
        auto t = std::make_shared<std::thread>(session, socket);
        // insert to set in case the for loop over the thread will end
        thread_set.insert(t);
    }
}

int main() {
    try {
        std::cout << "This is service!\n";
        boost::asio::io_context  ioc;
        server(ioc, 10086);
        for (auto& t : thread_set) {
            t->join();
        }
    } catch (std::exception& e) {
        std::cerr << "Exception " << e.what() << "\n";
    }
    return 0;
}
```

#ifndef ORANGE_TCP
#define ORANGE_TCP

#include <iostream>
#include <sys/types.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <string.h>
#include <string>
#include <iomanip> 


void send_msg(std::string msg, int sock){
    
    char buf[2048];
    int sendRes;

    int msg_length = msg.size();

    // send header
    std::ostringstream ss;
    ss << std::left << std::setfill(' ') << std::setw(64) << msg_length;
    auto padded{ ss.str() };
    sendRes = send(sock, padded.c_str(), 64, 0);
    if (sendRes == -1)
    {
        std::cout << "Could not send to server! Whoops!\r\n";
        return;
    }

    // send msg 
    sendRes = send(sock, msg.c_str(), msg_length, 0);
    if (sendRes == -1)
    {
        std::cout << "Could not send to server! Whoops!\r\n";
        return;
    }

    // receive message
    memset(buf, 0, 2048);
    int bytesReceived = recv(sock, buf, 2048, 0);
    if (bytesReceived == -1)
    {
        std::cout << "There was an error getting response from server\r\n";
    }
    else
    {
        std::cout << "[SERVER]> " << std::string(buf, bytesReceived) << "\r\n";
    }
}


int connect_and_send_coordinate(std::string str_ball_cord)
{
    std::cout << "start running..." << std::endl;

    //	Create a socket
    int sock = socket(AF_INET, SOCK_STREAM, 0);
        
    if (sock == -1)
    {
        return 1;
    }
    std::cout << "client created..." << std::endl;


    //	Create a hint structure for the server we're connecting with
    int port = 5050;
    std::string ipAddress = "127.0.1.1";

    sockaddr_in hint;
    hint.sin_family = AF_INET;
    hint.sin_port = htons(port);
    inet_pton(AF_INET, ipAddress.c_str(), &hint.sin_addr);

    //	Connect to the server on the socket
    int connectRes = connect(sock, (sockaddr*)&hint, sizeof(hint));
    if (connectRes == -1)
    {
        std::cout << "Fail to connect" << std::endl;
        return 1;
    }

    std::cout << "client connected..." << std::endl;

    send_msg(str_ball_cord, sock);
    
    // disconnect
    send_msg("!DISCONNECT", sock);

    //	Close the socket
    close(sock);
    return 0;
}


#endif
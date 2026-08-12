#ifndef _CONFIG_H
#define _CONFIG_H

// include the necessary libraries
#include "caf/all.hpp"
#include "caf/io/all.hpp"
#include <chrono>
#include <fstream>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <queue>
#include <cstdlib>  // For popen
#include <sstream>  // For std::ostringstream
#include <iostream> // For std::stod
#include <random>
#include <thread>
#include <atomic>


// using the necessary namespaces
using namespace caf;
using namespace std;

class config : public caf::actor_system_config
{
public:
    uint16_t port = 0;
    std::string host = "localhost";
    bool server_mode = false;
    std::string betas = "0.1,0.5,1.0,2.0,4.0"; // comma-separated beta values to sweep
    int workers = 3; // number of parallel worker actors the server spawns
    int gpus = 2; // number of local GPUs to round-robin worker training runs across (0 disables pinning)
    config()
    {
        // Add the port and host to the config
        opt_group{custom_options_, "global"}
            .add(port, "port,p", "set port")
            .add(host, "host,H", "set host (ignored in server mode)")
            .add(server_mode, "server-mode,s", "enable server mode")
            .add(betas, "betas,b", "comma-separated list of beta values to sweep")
            .add(workers, "workers,w", "number of parallel worker actors to spawn (server mode)")
            .add(gpus, "gpus,g", "number of local GPUs to round-robin worker training runs across (0 disables pinning)");
    }
};

// Parses a comma-separated list of doubles, e.g. "0.1,0.5,1.0" -> {0.1, 0.5, 1.0}
inline std::vector<double> parse_betas(const std::string& csv)
{
    std::vector<double> values;
    std::stringstream ss(csv);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            values.push_back(std::stod(token));
        }
    }
    return values;
}

#endif
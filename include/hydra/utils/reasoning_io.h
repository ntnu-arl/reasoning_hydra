#ifndef INCLUDE_HYDRA_UTILS_REASONING_IO_H_
#define INCLUDE_HYDRA_UTILS_REASONING_IO_H_
#endif  // INCLUDE_HYDRA_UTILS_REASONING_IO_H_

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "hydra/reasoning/reasoning_output.h"

namespace hydra {

using json = nlohmann::json;

void parseReasoningJson(const std::string& filename,
                        ReasoningOutput& data,
                        const std::string& room_name);

bool readLines(const std::string& filename, std::vector<std::string>& lines);
}  // namespace hydra

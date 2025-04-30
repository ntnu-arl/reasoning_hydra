/* -----------------------------------------------------------------------------
 * Copyright 2022 Massachusetts Institute of Technology.
 * All Rights Reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Research was sponsored by the United States Air Force Research Laboratory and
 * the United States Air Force Artificial Intelligence Accelerator and was
 * accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views
 * and conclusions contained in this document are those of the authors and should
 * not be interpreted as representing the official policies, either expressed or
 * implied, of the United States Air Force or the U.S. Government. The U.S.
 * Government is authorized to reproduce and distribute reprints for Government
 * purposes notwithstanding any copyright notation herein.
 * -------------------------------------------------------------------------- */
#pragma once
#include <config_utilities/virtual_config.h>

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "hydra/common/input_queue.h"
#include "hydra/common/module.h"
#include "hydra/input/data_receiver.h"
#include "hydra/input/input_packet.h"

namespace hydra {

struct PoseStatus {
  Eigen::Quaterniond target_R_source;
  Eigen::Vector3d target_p_source;
  bool is_valid = false;
  PoseStatus() = default;
  explicit PoseStatus(const bool& valid)
      : target_R_source(Eigen::Quaterniond::Identity()),
        target_p_source(Eigen::Vector3d::Zero()),
        is_valid(valid) {}
  PoseStatus(const Eigen::Quaterniond& R, const Eigen::Vector3d& p)
      : target_R_source(R), target_p_source(p), is_valid(true) {}
  operator bool() const { return is_valid; }

  Eigen::Matrix4d toMatrix() const {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = target_R_source.toRotationMatrix();
    T.block<3, 1>(0, 3) = target_p_source;
    return T;
  }

  std::string toString() const {
    return "Valid: " + std::to_string(is_valid) + "\n" + "Rotation: [" +
           std::to_string(target_R_source.x()) + ", " +
           std::to_string(target_R_source.y()) + ", " +
           std::to_string(target_R_source.z()) + ", " +
           std::to_string(target_R_source.w()) + "]\n" + "Translation: [" +
           std::to_string(target_p_source.x()) + ", " +
           std::to_string(target_p_source.y()) + ", " +
           std::to_string(target_p_source.z()) + "]\n";
  }

  inline operator Eigen::Isometry3d() const {
    return Eigen::Translation3d(target_p_source) * target_R_source;
  }

  Eigen::Isometry3d toIsometry() const {
    return Eigen::Translation3d(target_p_source) * target_R_source;
  }
};

class InputModule : public Module {
 public:
  using OutputQueue = InputQueue<InputPacket::Ptr>;
  struct Config {
    std::vector<config::VirtualConfig<DataReceiver>> receivers;
  } const config;

  InputModule(const Config& config, const OutputQueue::Ptr& output_queue);

  virtual ~InputModule();

  void start() override;

  void stop() override;

  void save(const LogSetup& log_setup) override;

  std::string printInfo() const override;

 protected:
  void dataSpin();

  void stopImpl();

  virtual PoseStatus getBodyPose(uint64_t timestamp) = 0;

 protected:
  OutputQueue::Ptr queue_;
  std::atomic<bool> should_shutdown_{false};

  std::vector<std::unique_ptr<DataReceiver>> receivers_;
  std::unique_ptr<std::thread> data_thread_;
};

void declare_config(InputModule::Config& config);

}  // namespace hydra

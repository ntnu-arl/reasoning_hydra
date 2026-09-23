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
#include <config_utilities/factory.h>

#include <optional>

#include "hydra/backend/update_functions.h"

namespace hydra {

struct UpdateAgentsFunctor : public UpdateFunctor {
  struct Config {
    bool enable_agent_keyframes = true;  // Enable adding agent keyframes
    float cos_sim_thresh = 0.4;  // Cosine similarity threshold for keyframe addition
    double min_translation_m =
        2.0;  // Min translation between agent nodes to add a keyframe
    double min_rotation_deg =
        20.0;  // Min rotation between agent nodes to add a keyframe
  } const config;

  explicit UpdateAgentsFunctor(const Config& config);

  void call(const DynamicSceneGraph&,
            SharedDsgInfo& graph,
            const UpdateInfo::ConstPtr& info) override;

  void updateAgentKeyframes(SharedDsgInfo& dsg, const UpdateInfo::ConstPtr& info);

 private:
  std::optional<NodeSymbol> last_keyframe_id_ = std::nullopt;
  inline static const auto registration_ =
      config::RegistrationWithConfig<UpdateFunctor, UpdateAgentsFunctor, Config>(
          "UpdateAgentsFunctor");
};

void declare_config(UpdateAgentsFunctor::Config&);

}  // namespace hydra

/* -----------------------------------------------------------------------------
 * BSD 3-Clause License
 *
 * Copyright (c) 2026, NTNU Autonomous Robots Lab
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * -------------------------------------------------------------------------- */
#pragma once
#include <config_utilities/virtual_config.h>

#include <Eigen/Geometry>
#include <memory>
#include <string>
#include <vector>

#include "hydra/frontend/place_2d_interface.h"
#include "hydra/voronoi/graph_extractor.h"

namespace hydra {

class GvdPlace2DExtractor : public Place2DInterface {
 public:
  struct Config {
    std::string layer = DsgLayers::TRAVERSABILITY;
    voronoi::GraphExtractor::Config extractor_config;
    std::vector<Sink::Factory> sinks;
  } const config;

  GvdPlace2DExtractor(const Config& config);

  virtual ~GvdPlace2DExtractor() = default;

  void detect(const ActiveWindowOutput& msg) override;

  void updateGraph(const ActiveWindowOutput& msg, DynamicSceneGraph& graph) override;

 protected:
  voronoi::GraphExtractor::Ptr graph_extractor_;
  Eigen::Isometry3d world_T_body_;
  std::set<NodeId> old_nodes_;
  std::set<std::pair<NodeId, NodeId>> old_edges_;
  Sink::List sinks_;

 private:
  inline static const auto registration_ =
      config::RegistrationWithConfig<Place2DInterface, GvdPlace2DExtractor, Config>(
          "gvd");
};

void declare_config(GvdPlace2DExtractor::Config& config);

}  // namespace hydra

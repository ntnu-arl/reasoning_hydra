#include "hydra/navigation/object_search_module.h"

namespace hydra {

void declare_config(ObjectSearchModule::Config& config) {
  using namespace config;
  name("ObjectSearchConfig");
}

ObjectSearchModule::ObjectSearchModule(const Config& config) : config(config) {
  input_queue_.reset(new InputQueue<ObjectSearchInput::Ptr>());
  output_queue_.reset(new InputQueue<ObjectSearchOutput::Ptr>());
}

ObjectSearchModule::~ObjectSearchModule() { stopImpl(); }

void ObjectSearchModule::start() {
  spin_thread_.reset(new std::thread(&ObjectSearchModule::spin, this));
  LOG(INFO) << "[Hydra ObjectSearch] started!";
}

void ObjectSearchModule::stopImpl() {
  should_shutdown_ = true;
  if (spin_thread_) {
    VLOG(2) << "[Hydra ObjectSearch] joining object search thread and stopping";
    spin_thread_->join();
    spin_thread_.reset();
    VLOG(2) << "[Hydra ObjectSearch] stopped!";
  }
}

void ObjectSearchModule::stop() { stopImpl(); }

void ObjectSearchModule::save(const LogSetup& log_setup) {}

std::string ObjectSearchModule::printInfo() const {
  std::stringstream ss;
  ss << config::toString(config);
  return ss.str();
}

void ObjectSearchModule::spin() {
  bool should_shutdown = false;
  while (!should_shutdown) {
    bool has_data = input_queue_->poll();
    if (GlobalInfo::instance().force_shutdown() || !has_data) {
      // copy over shutdown request
      should_shutdown = should_shutdown_;
    }

    if (!has_data) {
      continue;
    }

    spinOnce(input_queue_->front());
    input_queue_->pop();
  }
}

void ObjectSearchModule::spinOnce(const ObjectSearchInput::Ptr& input) {}

void ObjectSearchModule::setGraph(const DynamicSceneGraph::Ptr& scene_graph) {
  std::lock_guard<std::mutex> lock(mutex_);
  scene_graph_ = scene_graph;
}

}  // namespace hydra

#include "hydra/navigation/navigation_module.h"

namespace hydra {

void declare_config(NavigationModule::Config& config) {
  using namespace config;
  name("NavigationConfig");
}

NavigationModule::NavigationModule(const Config& config) : config(config) {
  navigation_queue_.reset(new InputQueue<NavigationInput::Ptr>());
}

NavigationModule::~NavigationModule() { stopImpl(); }

void NavigationModule::start() {
  spin_thread_.reset(new std::thread(&NavigationModule::spin, this));
  LOG(INFO) << "[Hydra Navigation] started!";
}

void NavigationModule::stopImpl() {
  should_shutdown_ = true;
  if (spin_thread_) {
    VLOG(2) << "[Hydra Navigation] joining navigation thread and stopping";
    spin_thread_->join();
    spin_thread_.reset();
    VLOG(2) << "[Hydra Navigation] stopped!";
  }
}

void NavigationModule::stop() { stopImpl(); }

void NavigationModule::save(const LogSetup& log_setup) {}

std::string NavigationModule::printInfo() const {
  std::stringstream ss;
  ss << config::toString(config);
  return ss.str();
}

void NavigationModule::spin() {
  bool should_shutdown = false;
  while (!should_shutdown) {
    bool has_data = navigation_queue_->poll();
    if (GlobalInfo::instance().force_shutdown() || !has_data) {
      // copy over shutdown request
      should_shutdown = should_shutdown_;
    }

    if (!has_data) {
      continue;
    }

    spinOnce(navigation_queue_->front());
    navigation_queue_->pop();
  }
}

void NavigationModule::spinOnce(const NavigationInput::Ptr& input) {
  NavigationOutput output;
}

}  // namespace hydra

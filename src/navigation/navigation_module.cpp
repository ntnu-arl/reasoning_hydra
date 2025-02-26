#include "hydra/navigation/navigation_module.h"

namespace hydra {

void declare_config(NavigationModule::Config& config) {
  using namespace config;
  name("NavigationConfig");
}

NavigationModule::NavigationModule(const Config& config) : config(config) {
  input_queue_.reset(new InputQueue<NavigationInput::Ptr>());
  output_queue_.reset(new InputQueue<NavigationOutput>());
  shortest_path_methods_["dijkstra"] =
      std::function<void(const std::map<NodeId, SceneGraphNode::Ptr>&,
                         const std::set<EdgeKey>&,
                         const NodeId&,
                         const NodeId&,
                         std::vector<NodeId>&,
                         std::vector<Eigen::Vector3d>&)>(dijkstra);
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

void NavigationModule::spinOnce(const NavigationInput::Ptr& input) {
  std::string method = input->method;
  if (shortest_path_methods_.count(method) == 0) {
    LOG(INFO) << "Method " << method << " not supported, Dijkstra will be used!";
    method = "dijkstra";
  }
  std::lock_guard<std::mutex> lock(mutex_);
  NavigationOutput output;
  const auto& place_nodes =
      scene_graph_->getLayer(spark_dsg::DsgLayers::PLACES).nodes();
  const auto& object_layer = scene_graph_->getLayer(spark_dsg::DsgLayers::OBJECTS);
  const auto& agent_layer =
      scene_graph_->dynamicLayersOfType(spark_dsg::DsgLayers::AGENTS).begin()->second;
  const auto& agent_node = agent_layer->getNodeByIndex(agent_layer->numNodes() - 1);
  const auto& graph_edges =
      scene_graph_->getLayer(spark_dsg::DsgLayers::PLACES).edges();
  std::set<EdgeKey> edges;
  for (const auto& [key, _] : graph_edges) {
    if (!place_nodes.count(key.k1) || !place_nodes.count(key.k2)) {
      continue;
    }
    edges.insert(key);
  }

  for (size_t i = 0; i < input->object_ids.size(); ++i) {
    const auto& [obj1, obj2] = input->object_ids[i];
    NavigationPath path;
    if (!findNavigation(obj1,
                        obj2,
                        method,
                        place_nodes,
                        object_layer,
                        agent_layer,
                        agent_node,
                        edges,
                        path)) {
      continue;
    }
    path.explanation = input->explanation[i];
    output.push_back(path);
  }
  if (output.empty()) {
    return;
  }
  output_queue_->push(output);
}

bool NavigationModule::findNavigation(const NodeId& obj1,
                                      const NodeId& obj2,
                                      const std::string& method,
                                      const SceneGraphLayer::Nodes& place_nodes,
                                      const SceneGraphLayer& object_layer,
                                      const DynamicSceneGraphLayer::Ptr& agent_layer,
                                      const SceneGraphNode& agent_node,
                                      const std::set<EdgeKey>& edges,
                                      NavigationPath& output) const {
  const auto& obj1_node = object_layer.findNode(obj1);
  const auto& obj2_node = object_layer.findNode(obj2);
  if (!obj1_node || !obj2_node) {
    LOG(ERROR) << "Object node not found!";
    return false;
  }
  if (obj1_node->parents().empty() || obj2_node->parents().empty()) {
    LOG(ERROR) << "Object node has no parent!";
    return false;
  }
  const auto& obj1_place = *obj1_node->parents().begin();
  const auto& obj2_place = *obj2_node->parents().begin();
  if (!place_nodes.count(obj1_place) || !place_nodes.count(obj2_place)) {
    LOG(ERROR) << "Object place node not found!";
    return false;
  }

  const auto& agent_place_id = agent_node.getParent();
  if (!agent_place_id) {
    LOG(ERROR) << "Agent node has no parent!";
    return false;
  }

  if (!place_nodes.count(*agent_place_id)) {
    LOG(ERROR) << "Agent place node not found!";
    return false;
  }

  output.object_id = obj1;
  output.target_id = obj2;
  output.object_label = obj1_node->attributes<SemanticNodeAttributes>().name;
  output.target_label = obj2_node->attributes<SemanticNodeAttributes>().name;
  std::vector<NodeId> agent_obj1_path;
  std::vector<Eigen::Vector3d> agent_obj1_path_points;
  std::visit(
      [&](auto&& func) {
        return func(place_nodes,
                    edges,
                    *agent_place_id,
                    obj1_place,
                    agent_obj1_path,
                    agent_obj1_path_points);
      },
      shortest_path_methods_.at(method));
  std::vector<NodeId> obj1_obj2_path;
  std::vector<Eigen::Vector3d> obj1_obj2_path_points;
  std::visit(
      [&](auto&& func) {
        return func(place_nodes,
                    edges,
                    obj1_place,
                    obj2_place,
                    obj1_obj2_path,
                    obj1_obj2_path_points);
      },
      shortest_path_methods_.at(method));

  agent_obj1_path_points.emplace(agent_obj1_path_points.begin(),
                                 agent_node.attributes<NodeAttributes>().position);
  agent_obj1_path_points.push_back(obj1_node->attributes<NodeAttributes>().position);
  obj1_obj2_path_points.emplace(obj1_obj2_path_points.begin(),
                                obj1_node->attributes<NodeAttributes>().position);
  obj1_obj2_path_points.push_back(obj2_node->attributes<NodeAttributes>().position);

  output.agent_to_target = agent_obj1_path_points;
  output.target_to_object = obj1_obj2_path_points;
  return true;
}

void NavigationModule::setGraph(const DynamicSceneGraph::Ptr& scene_graph) {
  std::lock_guard<std::mutex> lock(mutex_);
  scene_graph_ = scene_graph;
}

}  // namespace hydra

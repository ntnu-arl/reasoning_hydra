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

#include "hydra/voronoi/segment_expander.h"

namespace hydra {
namespace voronoi {

SegmentExpander::SegmentExpander() {}

void SegmentExpander::Initialize(cv::Mat& _map,
                                 cv::Mat& _dist_field,
                                 cv::Mat& _voronoi_path) {
  nx_ = _map.cols;
  ny_ = _map.rows;
  ns_ = nx_ * ny_;

  distance_field_.reset(new float[nx_ * ny_]);
  voronoi_graph_.reset(new int8_t[nx_ * ny_]);
  global_map_.reset(new int8_t[nx_ * ny_]);

  float* distance_field = distance_field_.get();
  int8_t* voronoi_graph = voronoi_graph_.get();
  int8_t* global_map = global_map_.get();

  for (int i = 0; i < ns_; i++) {
    distance_field[i] = ((float*)_dist_field.data)[i];
    voronoi_graph[i] = ((int8_t*)_voronoi_path.data)[i];
    global_map[i] = ((int8_t*)_map.data)[i];
  }
}

void SegmentExpander::optimizeSegments(std::vector<Segment>& _segments,
                                       float _max_pixels_crossing,
                                       float _max_pixels_end_segment) {
  // Move all segments with distance less than _maxPixelCrossing into the crossing
  for (uint32_t i = 0; i < _segments.size(); i++) {
    if (!_segments[i].getOptStart()) {
      optimizeSegmentsAroundPoint(
          _segments, _segments[i].getStart(), _max_pixels_crossing, i);
    }

    if (!_segments[i].getOptEnd()) {
      optimizeSegmentsAroundPoint(
          _segments, _segments[i].getEnd(), _max_pixels_crossing, i);
    }
  }

  // Remove segments with zero length...
  for (uint32_t i = 0; i < _segments.size(); i++) {
    if ((_segments[i].getStart() - _segments[i].getEnd()).norm() == 0) {
      removeSegmentFromList(i, _segments);
      // if we erase one segment we have to check the same spot again...
      i--;
    }
  }

  // Important do this only in the end of the optimization
  for (uint32_t i = 0; i < _segments.size(); i++) {
    // Its important to have a sorted segment list by id... Therefore we need to update
    // all segments when we remove one of them As well we need to remove all existing
    // neighbors before erasing...
    if ((_segments[i].getStart() - _segments[i].getEnd()).norm() <
            _max_pixels_end_segment &&
        (_segments[i].getPredecessors().size() == 0 ||
         _segments[i].getSuccessors().size() == 0)) {
      removeSegmentFromList(i, _segments);
      i--;
    }
  }
}

void SegmentExpander::removeSegmentFromList(const uint32_t _id,
                                            std::vector<Segment>& _segments) {
  // Remove neighbors
  for (uint32_t pd : _segments[_id].getPredecessors()) {
    _segments[pd].cleanNeighbors(_id);
  }
  for (uint32_t sc : _segments[_id].getSuccessors()) {
    _segments[sc].cleanNeighbors(_id);
  }
  _segments.erase(_segments.begin() + _id);
  // Reorder graph
  for (uint32_t j = 0; j < _segments.size(); j++) {
    _segments[j].decreaseNeighborIdAbove(_id);
  }
}

void SegmentExpander::optimizeSegmentsAroundPoint(std::vector<Segment>& _segments,
                                                  const Eigen::Vector2d& pt,
                                                  float max_pixels,
                                                  int start_index) {
  // References to segments of a crossing
  std::vector<Segment*> connectedSegmnetsStart;
  std::vector<Segment*> connectedSegmnetsEnd;

  // find all segments to optimize
  for (uint32_t i = start_index; i < _segments.size(); i++) {
    if (!_segments[i].getOptStart()) {
      if ((_segments[i].getStart() - pt).norm() < max_pixels) {
        connectedSegmnetsStart.push_back(&_segments[i]);
        _segments[i].setStart(pt);
        _segments[i].getOptStart() = true;
      }
    }

    if (!_segments[i].getOptEnd()) {
      if ((_segments[i].getEnd() - pt).norm() < max_pixels) {
        connectedSegmnetsEnd.push_back(&_segments[i]);
        _segments[i].setEnd(pt);
        _segments[i].getOptEnd() = true;
      }
    }
  }

  // Connect all optimized segments
  for (uint32_t i = 0; i < connectedSegmnetsStart.size(); i++) {
    for (uint32_t j = 0; j < connectedSegmnetsStart.size(); j++) {
      if (i != j) {
        if (!(*connectedSegmnetsStart[i])
                 .containsPredecessor((*connectedSegmnetsStart[j]).getId())) {
          (*connectedSegmnetsStart[i])
              .addPredecessor((*connectedSegmnetsStart[j]).getId());
        }
      }
    }

    for (uint32_t j = 0; j < connectedSegmnetsEnd.size(); j++) {
      if (!(*connectedSegmnetsStart[i])
               .containsPredecessor((*connectedSegmnetsEnd[j]).getId())) {
        (*connectedSegmnetsStart[i]).addPredecessor((*connectedSegmnetsEnd[j]).getId());
      }
    }
  }

  for (uint32_t i = 0; i < connectedSegmnetsEnd.size(); i++) {
    for (uint32_t j = 0; j < connectedSegmnetsStart.size(); j++) {
      if (!(*connectedSegmnetsEnd[i])
               .containsSuccessor((*connectedSegmnetsStart[j]).getId())) {
        (*connectedSegmnetsEnd[i]).addSuccessor((*connectedSegmnetsStart[j]).getId());
      }
    }

    for (uint32_t j = 0; j < connectedSegmnetsEnd.size(); j++) {
      if (i != j) {
        if (!(*connectedSegmnetsEnd[i])
                 .containsSuccessor((*connectedSegmnetsEnd[j]).getId())) {
          (*connectedSegmnetsEnd[i]).addSuccessor((*connectedSegmnetsEnd[j]).getId());
        }
      }
    }
  }
}

const std::vector<Segment> SegmentExpander::splitPath(
    const std::vector<Eigen::Vector2d>& _path, const float _minimum_length) {
  std::vector<Segment> segments;

  if (_path.size() > 1 && _path.size() <= _minimum_length) {
    float min_seg_width = getMinSegmentWidth(_path);
    Segment seg(_path, 2 * min_seg_width);
    segments.push_back(seg);
  } else if (_path.size() > 1 && _path.size() > _minimum_length) {
    uint32_t nr_segments = _path.size() / _minimum_length;
    float real_length = std::ceil((float)_path.size() / (float)nr_segments);

    for (uint32_t i = 0; i < nr_segments; i++) {
      // Segment
      std::vector<Eigen::Vector2d> map_path_segment;
      for (uint32_t j = std::max<int>(0, i * real_length);
           j <= (i + 1) * real_length && j < _path.size();
           j++) {
        map_path_segment.push_back({_path[j][0], _path[j][1]});
      }

      float min_seg_width = getMinSegmentWidth(map_path_segment);
      Segment newSegment(std::vector<Eigen::Vector2d>(map_path_segment),
                         2 * min_seg_width);

      // Push back and add neighbors
      segments.push_back(newSegment);
      if (i > 0) {
        segments[segments.size() - 2].addSuccessor(
            segments[segments.size() - 1].getId());
        segments[segments.size() - 1].addPredecessor(
            segments[segments.size() - 2].getId());
      }
    }
  }

  return segments;
}

std::vector<Segment> SegmentExpander::getGraph(
    const std::vector<std::vector<Eigen::Vector2d>>& _endPoints,
    float* _potential,
    const float _min_length,
    float _optimize_pixels_crossing,
    const float _optimize_pixels_end_segments) {
  Segment::resetId();

  // We need at least a 10th of max length for seg optimization to find unconnected
  // edges from skeletonizing
  _optimize_pixels_crossing =
      std::max<float>(_min_length / 10, _optimize_pixels_crossing);

  // To be able to change the segments in the crossing class it has to be somehow
  // accessable over the heap -> shared_ptr
  std::shared_ptr<std::vector<Segment>> segments =
      std::make_shared<std::vector<Segment>>();
  std::vector<std::vector<Eigen::Vector2d>> endPoints = _endPoints;

  // Convert endpoints to crossings
  std::vector<Crossing> crossings_;
  for (uint32_t i = 0; i < endPoints.size(); i++) {
    Crossing c(std::vector<Eigen::Vector2d>(endPoints[i]));
    c.setSegmentReference(segments);
    crossings_.push_back(c);
  }

  // Find all paths connecting the crossings
  while (endPoints.size() > 0) {
    std::vector<Eigen::Vector2d> startCrossing = endPoints.back();
    if (startCrossing.size() > 0) {
      Eigen::Vector2d startPoint = startCrossing.back();

      // Reset the potential of each crossing endpoint to prepare for the expansion
      for (const Eigen::Vector2d& cVec : startCrossing) {
        Index idx(cVec[0], cVec[1], nx_, 0, 0, 0);
        _potential[idx.i] = 0;
      }

      Index start(startPoint[0], startPoint[1], nx_, 0, 0, 0);

      std::vector<Eigen::Vector2d> path = getPath(start, _potential, endPoints);

      std::vector<Segment> pathSegments = splitPath(path, _min_length);
      segments->insert(segments->end(), pathSegments.begin(), pathSegments.end());
      removeEndpoint(start, endPoints);
    } else {
      endPoints.pop_back();
    }
  }

  // Find the correct crossing for each segment
  for (Segment& s : (*segments)) {
    for (Crossing& c : crossings_) {
      c.tryAddSegment(s);
    }
  }

  // Optimize Crossings by moving the center and removing unecessary "double crossings"
  optimizeSegments(
      (*segments), _optimize_pixels_crossing, _optimize_pixels_end_segments);

  return (*segments);
}

float SegmentExpander::getMinSegmentWidth(const std::vector<Eigen::Vector2d>& _path) {
  if (_path.size() == 0) return 0;

  Index p1(_path[0][0], _path[0][1], nx_, 0, 0, 0);
  float minimum_path_space = distance_field_[p1.i];

  for (const Eigen::Vector2d& point :
       _path) {  // auto it = _path.begin(); it != _path.end(); ++it)
    Index p(point[0], point[1], nx_, 0, 0, 0);
    minimum_path_space = std::min<float>(distance_field_[p.i], minimum_path_space);
  }

  return minimum_path_space;
}

std::vector<std::vector<Eigen::Vector2d>> SegmentExpander::calcEndpoints(
    float* _potential) {
  std::vector<std::vector<Eigen::Vector2d>> endpoints;

  std::fill(_potential, _potential + ns_, -1);

  for (int i = 0; i < ns_; i++) {
    // dont examine allready found potentials
    if (_potential[i] >= 0) continue;

    // only look for voronoi points
    if (voronoi_graph_[i] >= 0) continue;

    if (nrOfNeighbours(i) > 2) {
      std::vector<Eigen::Vector2d> crossing =
          expandCrossing(Index(i, 0, 0, 0), _potential);
      endpoints.push_back(crossing);
    }
  }

  return std::vector<std::vector<Eigen::Vector2d>>(endpoints);
}

bool SegmentExpander::checkSegmentPoint(const SegmentExpander::Index& _point) {
  // Check Boundries
  if (_point.i < 0 || _point.i > ns_) return false;

  if (global_map_[_point.i] > 0) return false;

  if (voronoi_graph_[_point.i] >= 0) return false;

  if (nrOfNeighbours(_point.i) != 2) return false;

  return true;
}

std::vector<Eigen::Vector2d> SegmentExpander::getPath(
    const SegmentExpander::Index& _start,
    float* _potential,
    const std::vector<std::vector<Eigen::Vector2d>>& _endpoints) {
  std::vector<Eigen::Vector2d> points;

  int cycle = 0;
  int cycles = ns_;
  _potential[_start.i] = _start.potential;

  Index current(_start.i, _potential[_start.i], 0, _potential[_start.i]);
  // min_d = distance_field_[_start.i];
  // clear the queue
  clearpq(queue_);
  queue_.push(current);

  while (!(queue_.empty()) && (cycle < cycles)) {
    if (queue_.empty()) break;

    current = queue_.top();
    queue_.pop();

    // min_d = std::min<float>(min_d, distance_field_[current.i]);
    points.push_back({current.getX(nx_), current.getY(nx_)});

    if (current.i != _start.i && isEndpoint(current, _endpoints)) {
      Eigen::Vector2d point;
      point[0] = current.getX(nx_);
      point[1] = current.getY(nx_);
      return points;
    } else {
      addExpansionCandidate(current, current.offsetDist(1, 0, nx_, ny_), _potential);
      addExpansionCandidate(current, current.offsetDist(0, 1, nx_, ny_), _potential);
      addExpansionCandidate(current, current.offsetDist(-1, 0, nx_, ny_), _potential);
      addExpansionCandidate(current, current.offsetDist(0, -1, nx_, ny_), _potential);

      addExpansionCandidate(current, current.offsetDist(1, 1, nx_, ny_), _potential);
      addExpansionCandidate(current, current.offsetDist(-1, 1, nx_, ny_), _potential);
      addExpansionCandidate(current, current.offsetDist(-1, -1, nx_, ny_), _potential);
      addExpansionCandidate(current, current.offsetDist(1, -1, nx_, ny_), _potential);
    }

    cycle++;
  }

  float dx = points.front()[0] - points.back()[0];
  float dy = points.front()[1] - points.back()[1];

  if (!find_edge_segments_ || points.size() <= 3 || std::sqrt(dx * dx + dy * dy) <= 3) {
    points.clear();
  }

  return points;
}

Eigen::Vector2d SegmentExpander::expandSegment(
    SegmentExpander::Index _start,
    float* _potential,
    const std::vector<std::vector<Eigen::Vector2d>>& _endpoints) {
  std::vector<Eigen::Vector2d> points;

  int cycle = 0;
  int cycles = ns_;
  _potential[_start.i] = _start.potential;

  Index current(_start.i, _potential[_start.i], 0, _potential[_start.i]);

  // clear the queue
  clearpq(queue_);
  queue_.push(current);

  while (!(queue_.empty()) && (cycle < cycles)) {
    if (queue_.empty()) break;

    current = queue_.top();
    queue_.pop();

    if (current.i != _start.i && isEndpoint(current, _endpoints)) {
      Eigen::Vector2d point;
      point[0] = current.getX(nx_);
      point[1] = current.getY(nx_);
      return point;
    } else {
      addExpansionCandidate(current, current.offsetDist(1, 0, nx_, ny_), _potential);
      addExpansionCandidate(current, current.offsetDist(0, 1, nx_, ny_), _potential);
      addExpansionCandidate(current, current.offsetDist(-1, 0, nx_, ny_), _potential);
      addExpansionCandidate(current, current.offsetDist(0, -1, nx_, ny_), _potential);

      addExpansionCandidate(current, current.offsetDist(1, 1, nx_, ny_), _potential);
      addExpansionCandidate(current, current.offsetDist(-1, 1, nx_, ny_), _potential);
      addExpansionCandidate(current, current.offsetDist(-1, -1, nx_, ny_), _potential);
      addExpansionCandidate(current, current.offsetDist(1, -1, nx_, ny_), _potential);
    }

    cycle++;
  }

  if (find_edge_segments_)
    return Eigen::Vector2d(current.getX(nx_), current.getY(nx_));
  else
    return Eigen::Vector2d(_start.getX(nx_), _start.getY(nx_));
}

std::vector<Eigen::Vector2d> SegmentExpander::expandCrossing(const Index& _start,
                                                             float* _potential) {
  std::vector<Eigen::Vector2d> points;

  int cycle = 0;
  int cycles = ns_;
  _potential[_start.i] = _start.potential;

  Index current(_start.i, _potential[_start.i], 0, _potential[_start.i]);

  // clear the queue
  clearpq(queue_);
  queue_.push(current);

  while (!(queue_.empty()) && (cycle < cycles)) {
    if (queue_.empty()) break;

    current = queue_.top();
    queue_.pop();

    if (nrOfNeighbours(current.i) <= 2) {
      Eigen::Vector2d point;
      point[0] = current.getX(nx_);
      point[1] = current.getY(nx_);
      points.push_back(point);
      _potential[current.i] = -2;  // Reset potential to find point
    } else {
      addExpansionCandidate(current, current.offsetDist(1, 0, nx_, ny_), _potential);
      addExpansionCandidate(current, current.offsetDist(0, 1, nx_, ny_), _potential);
      addExpansionCandidate(current, current.offsetDist(-1, 0, nx_, ny_), _potential);
      addExpansionCandidate(current, current.offsetDist(0, -1, nx_, ny_), _potential);

      addExpansionCandidate(current, current.offsetDist(1, 1, nx_, ny_), _potential);
      addExpansionCandidate(current, current.offsetDist(-1, 1, nx_, ny_), _potential);
      addExpansionCandidate(current, current.offsetDist(-1, -1, nx_, ny_), _potential);
      addExpansionCandidate(current, current.offsetDist(1, -1, nx_, ny_), _potential);
    }

    cycle++;
  }

  return points;
}

void SegmentExpander::addExpansionCandidate(const Index& current,
                                            const Index& next,
                                            float* potential) {
  float potentialPrev = potential[current.i];

  // Check Boundries
  if (next.i < 0 || next.i > ns_) return;

  // Dont update allready found potentials
  if (potential[next.i] >= 0) return;

  if (global_map_[next.i] > 0) return;

  if (voronoi_graph_[next.i] >= 0) return;

  float pot = potentialPrev + 1;
  float dist = 0;
  float weight = pot;

  potential[next.i] = pot;

  queue_.push(Index(next.i, weight, dist, pot));
}

uint32_t SegmentExpander::nrOfNeighbours(uint32_t i) const {
  // 0 1 2
  // 7   3
  // 6 5 4
  uint32_t neighbours[8] = {
      (voronoi_graph_[(Index(i, 0, 0, 0).offsetDist(-1, 1, nx_, ny_)).i] < 0),
      (voronoi_graph_[(Index(i, 0, 0, 0).offsetDist(0, 1, nx_, ny_)).i] < 0),
      (voronoi_graph_[(Index(i, 0, 0, 0).offsetDist(1, 1, nx_, ny_)).i] < 0),
      (voronoi_graph_[(Index(i, 0, 0, 0).offsetDist(1, 0, nx_, ny_)).i] < 0),
      (voronoi_graph_[(Index(i, 0, 0, 0).offsetDist(1, -1, nx_, ny_)).i] < 0),
      (voronoi_graph_[(Index(i, 0, 0, 0).offsetDist(0, -1, nx_, ny_)).i] < 0),
      (voronoi_graph_[(Index(i, 0, 0, 0).offsetDist(-1, -1, nx_, ny_)).i] < 0),
      (voronoi_graph_[(Index(i, 0, 0, 0).offsetDist(-1, 0, nx_, ny_)).i] < 0)};

  // Remove double neighbours e.g.:
  //   x x 0
  //   0 x 0
  //   0 x x
  // should result in 2

  uint32_t num_neighbours = neighbours[0] + neighbours[1] + neighbours[2] +
                            neighbours[3] + neighbours[4] + neighbours[5] +
                            neighbours[6] + neighbours[7];

  if (neighbours[0] && neighbours[1] && !neighbours[2]) num_neighbours--;

  if (!neighbours[0] && neighbours[1] && neighbours[2]) num_neighbours--;

  if (neighbours[2] && neighbours[3] && !neighbours[4]) num_neighbours--;

  if (!neighbours[2] && neighbours[3] && neighbours[4]) num_neighbours--;

  if (neighbours[4] && neighbours[5] && !neighbours[6]) num_neighbours--;

  if (!neighbours[4] && neighbours[5] && neighbours[6]) num_neighbours--;

  if (neighbours[6] && neighbours[7] && !neighbours[0]) num_neighbours--;

  if (!neighbours[6] && neighbours[7] && neighbours[0]) num_neighbours--;

  return num_neighbours;
}

void SegmentExpander::removeEndpoint(
    const SegmentExpander::Index& _current,
    std::vector<std::vector<Eigen::Vector2d>>& _endpoints) {
  for (uint32_t j = 0; j < _endpoints.size(); j++) {
    if (_endpoints[j].size() > 0) {
      for (uint32_t i = 0; i < _endpoints[j].size(); i++) {
        if ((int)(_endpoints[j][i][0]) == _current.getX(nx_) &&
            (int)(_endpoints[j][i][1]) == _current.getY(nx_)) {
          _endpoints[j].erase(_endpoints[j].begin() + i);

          if (_endpoints[j].size() == 0) {
            _endpoints.erase(_endpoints.begin() + j);
          }
        }
      }
    } else {
      _endpoints.erase(_endpoints.begin() + j);
    }
  }
}

bool SegmentExpander::isEndpoint(
    SegmentExpander::Index& _current,
    const std::vector<std::vector<Eigen::Vector2d>>& _endpoints) {
  for (auto it_crossing = _endpoints.begin(); it_crossing != _endpoints.end();
       it_crossing++) {
    for (auto it = it_crossing->begin(); it != it_crossing->end(); it++) {
      if ((*it)[0] == _current.getX(nx_) && (*it)[1] == _current.getY(nx_)) {
        return true;
      }
    }
  }

  return false;
}

}  // namespace voronoi
}  // namespace hydra

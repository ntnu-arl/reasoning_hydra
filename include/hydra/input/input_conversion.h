#pragma once

#include <memory>

#include "hydra/input/input_data.h"

namespace hydra {

struct InputPacket;

namespace conversions {

/**
 * @brief Convert an input packet to a valid input data format if possible.
 * @param input_packet The input packet to convert.
 * @param vertices_in_world_frame If true, convert the vertex image to be in world
 * frame. Otherwise it will be in sensor frame.
 * @return The input data if successful, nullptr otherwise.
 */
std::unique_ptr<InputData> parseInputPacket(const InputPacket& input_packet,
                                            const bool vertices_in_world_frame = false);

// TODO(lschmid): Ported this from the input data. Does not seem to be used anywhere
// though.
bool hasSufficientData(const InputData& data);

/**
 * @brief make sure that all the images are of the right type
 */
bool normalizeData(InputData& data, bool normalize_labels = true);

bool normalizeDepth(InputData& data);

bool colorToLabels(cv::Mat& label_image, const cv::Mat& colors);

bool convertLabels(InputData& data);

bool convertDepth(InputData& data);

bool convertColor(InputData& data);

/**
 * @brief Ensure that the vertex map is in the correct frame.
 * @param data The input data to convert.
 * @param in_world_frame If true, the vertex map will be converted to world frame.
 * Otherwise it will be in sensor frame.
 */
void convertVertexMap(InputData& data, bool in_world_frame);

class RvlCodec {
 public:
  RvlCodec();
  // Compress input data into output. The size of output can be equal to
  // (1.5 * numPixels + 4) in the worst case.
  int CompressRVL(const unsigned short* input, unsigned char* output, int numPixels);
  // Decompress input data into output. The size of output must be
  // equal to numPixels.
  void DecompressRVL(const unsigned char* input, unsigned short* output, int numPixels);

 private:
  RvlCodec(const RvlCodec&);
  RvlCodec& operator=(const RvlCodec&);

  void EncodeVLE(int value);
  int DecodeVLE();

  int* buffer_;
  int* pBuffer_;
  int word_;
  int nibblesWritten_;
};

}  // namespace conversions
}  // namespace hydra

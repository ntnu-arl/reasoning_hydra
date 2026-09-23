target_sources(
  ${PROJECT_NAME}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/crossing.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/graph_extractor.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/segment_expander.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/segment.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/thinning.cpp
)

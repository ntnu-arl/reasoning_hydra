target_sources(
  ${PROJECT_NAME}
  PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/cos_sim_search.cpp
  PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/dijkstra.cpp
           ${CMAKE_CURRENT_SOURCE_DIR}/navigation_module.cpp
           ${CMAKE_CURRENT_SOURCE_DIR}/object_search_module.cpp
)

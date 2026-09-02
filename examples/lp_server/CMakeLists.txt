add_executable(parakeet-lp-server
    main.cpp
    ${CMAKE_SOURCE_DIR}/examples/server/openai_format.cpp)
target_link_libraries(parakeet-lp-server PRIVATE parakeet)
target_include_directories(parakeet-lp-server PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/examples/server)
find_package(Threads REQUIRED)
target_link_libraries(parakeet-lp-server PRIVATE Threads::Threads)
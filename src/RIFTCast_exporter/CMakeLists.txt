file(GLOB source "*.h*" "*.c*") 

get_filename_component(TARGET_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
ATCG_add_executable(${TARGET_NAME} target_name "${source}")

if(TARGET ${TARGET_NAME})
    target_link_libraries(${TARGET_NAME} PUBLIC RIFTCast_lib)
endif()
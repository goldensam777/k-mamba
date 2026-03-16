#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "optimatrixoptimatrix-cpu" for configuration ""
set_property(TARGET optimatrixoptimatrix-cpu APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(optimatrixoptimatrix-cpu PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "C"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/liboptimatrix-cpu.a"
  )

list(APPEND _cmake_import_check_targets optimatrixoptimatrix-cpu )
list(APPEND _cmake_import_check_files_for_optimatrixoptimatrix-cpu "${_IMPORT_PREFIX}/lib/liboptimatrix-cpu.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)

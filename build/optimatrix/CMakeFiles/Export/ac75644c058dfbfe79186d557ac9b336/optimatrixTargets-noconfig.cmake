#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "optimatrix::optimatrix" for configuration ""
set_property(TARGET optimatrix::optimatrix APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(optimatrix::optimatrix PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "ASM_NASM;C"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/liboptimatrix.a"
  )

list(APPEND _cmake_import_check_targets optimatrix::optimatrix )
list(APPEND _cmake_import_check_files_for_optimatrix::optimatrix "${_IMPORT_PREFIX}/lib/liboptimatrix.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)

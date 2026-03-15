#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "k-mamba::k-mamba" for configuration ""
set_property(TARGET k-mamba::k-mamba APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(k-mamba::k-mamba PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "C"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libk-mamba.a"
  )

list(APPEND _cmake_import_check_targets k-mamba::k-mamba )
list(APPEND _cmake_import_check_files_for_k-mamba::k-mamba "${_IMPORT_PREFIX}/lib/libk-mamba.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)

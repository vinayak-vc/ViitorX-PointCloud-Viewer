# Post-build step of pfview: ship the pf360 Python worker beside the exe
# (<exe dir>/pf360/pf360). Invoked as
#   cmake -DSRC=<repo>/tools/pf360/pf360 -DDST=<exe dir>/pf360/pf360 -P copy_worker.cmake
#
# A developer may replace <exe dir>/pf360 with a symbolic link / junction to the
# repo's tools/pf360 so worker edits take effect without a rebuild. Then DST
# resolves to SRC and copying would write the worker onto itself, so it is skipped.
if(NOT SRC OR NOT DST)
    message(FATAL_ERROR "copy_worker.cmake: SRC and DST are required")
endif()
get_filename_component(_src "${SRC}" REALPATH)
get_filename_component(_dst "${DST}" REALPATH)
if(_src STREQUAL _dst)
    message(STATUS "pf360 worker: ${DST} links to ${_src}; not copying")
    return()
endif()
file(MAKE_DIRECTORY "${DST}")
file(COPY "${SRC}/" DESTINATION "${DST}" PATTERN "__pycache__" EXCLUDE)
# Stale bytecode from an older copy would shadow nothing (sources are newer) but is clutter.
file(REMOVE_RECURSE "${DST}/__pycache__" "${DST}/ns_helpers/__pycache__")
